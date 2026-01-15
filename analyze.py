#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import cv2
import queue
import threading
from functools import partial
import math
import argparse
import matplotlib
matplotlib.use('Agg') # Non-interactive backend to avoid Qt/X11 issues
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Path Setup for Hailo Apps ---
# We need to import hailo_apps modules. Assuming we are in deadlift-analyzer/
current_dir = os.path.dirname(os.path.abspath(__file__))
hailo_apps_dir = os.path.join(current_dir, "hailo-apps")

# Add paths for imports
# utils.py is in .../standalone_apps/pose_estimation/
pose_utils_path = os.path.join(hailo_apps_dir, "hailo_apps", "python", "standalone_apps", "pose_estimation")
# core is in .../python/core (or common inside it)
core_path = os.path.join(hailo_apps_dir, "hailo_apps", "python", "core")

sys.path.append(pose_utils_path)
sys.path.append(hailo_apps_dir) # to allow 'from hailo_apps...'
sys.path.append(core_path)

# Try imports
try:
    from pose_estimation_utils import PoseEstPostProcessing
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
except ImportError:
    # Fallback if structure is slightly different (e.g. installed package)
    try:
        from common.hailo_inference import HailoInfer
    except ImportError:
        print("Error: Could not import HailoInfer. Check hailo-apps path.")
        sys.exit(1)

# --- Constants & Configuration ---
DEFAULT_MODEL_S = "/usr/local/hailo/resources/models/hailo8l/yolov8s_pose.hef"
DEFAULT_MODEL_M = "/usr/local/hailo/resources/models/hailo8l/yolov8m_pose.hef"
CONF_THRESHOLD = 0.3
NUM_KEYPOINTS = 17

# Keypoint Indices (COCO format)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Deadlift States
STATE_IDLE = "Standing / Start"
STATE_ECCENTRIC = "Eccentric (Descending)"
STATE_BOTTOM = "Bottom"
STATE_CONCENTRIC = "Concentric (Ascending)"
STATE_LOCKOUT = "Lockout"

# --- Geometry Helpers ---
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(p1, p2, p3):
    # Calculate angle at p2 (p1-p2-p3)
    # Vectors p2->p1 and p2->p3
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 * mag2 == 0:
        return 0
    
    angle_rad = math.acos(max(min(dot / (mag1 * mag2), 1.0), -1.0))
    return math.degrees(angle_rad)

def calculate_iou(box1, box2):
    # box: [ymin, xmin, ymax, xmax]
    # Determine intersection rectangle
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    
    # Area of intersection
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Area of union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0: return 0
    if union_area == 0: return 0
    return inter_area / union_area

# --- Video Writer Thread (Pipeline) ---
class AsyncVideoWriter:
    def __init__(self, path, fourcc, fps, size):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.queue = queue.Queue(maxsize=30) # Buffer 1 sec approx
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def write(self, frame):
        if not self.running: return
        # We must copy the frame if the upstream reuses the buffer, 
        # but cv2.read() usually allocates new. 
        # To be purely safe against race conditions if we modified it later (we don't):
        self.queue.put(frame) 
        
    def _run(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue
                
    def release(self):
        self.running = False
        self.thread.join()
        self.writer.release()

class DeadliftAnalyzer:
    def __init__(self):
        self.state = STATE_IDLE
        self.reps = 0
        self.errors = [] # List of errors for current rep
        self.all_errors = [] # History of (frame_idx, error_msg)
        
        self.start_height = None # Hip height at start
        self.start_sh_dist = None # Shoulder-Hip distance at start
        
        # Timing
        self.eccentric_start_time = 0
        self.concentric_start_time = 0
        self.bottom_time = 0
        
        # History for smoothing
        self.kp_history = [] 
        self.smooth_window = 5
        
        # Graphs data
        self.history_shk_angles = [] # (frame, angle)
        self.history_sh_dist = [] # (frame, dist)
        self.history_sh_dist = [] # (frame, dist)
        self.history_sh_dist = [] # (frame, dist)
        self.history_hip_traj = [] # (frame, x, y)
        self.frame_count = 0
        
        self.rep_history = {} # {rep_num: [error_msgs]}
        self.rep_timings = {} # {rep_num: {'asc': float, 'des': float}}

    def log_error(self, msg):
        # Current rep is self.reps + 1
        curr = self.reps + 1
        
        # Add to current frame errors (for return)
        if msg not in self.errors:
            self.errors.append(msg)
            
        # Add to all_errors (Legacy/Debug)
        self.all_errors.append((self.frame_count, f"Rep {curr}: {msg}"))
        
        # Add to Rep History
        if curr not in self.rep_history:
            self.rep_history[curr] = []
        
        if msg not in self.rep_history[curr]:
            self.rep_history[curr].append(msg)

    def smooth_keypoints(self, keypoints):
        # keypoints: (17, 3) [x, y, conf]
        self.kp_history.append(keypoints)
        if len(self.kp_history) > self.smooth_window:
            self.kp_history.pop(0)
        
        # Average x, y. Conf takes min or avg? Let's take avg.
        avg_kp = np.mean(np.array(self.kp_history), axis=0)
        return avg_kp

    def get_side_points(self, kps):
        # Check confidence of left vs right side
        # Side indices: Left (5,7,9,11,13,15), Right (6,8,10,12,14,16)
        left_conf = np.mean(kps[[5,7,9,11,13,15], 2])
        right_conf = np.mean(kps[[6,8,10,12,14,16], 2])
        
        # Return side, S, H, K, E (No Ankle/Wrist)
        if left_conf > right_conf:
            side = "Left"
            # Shoulder=5, Hip=11, Knee=13, Elbow=7
            return side, kps[5], kps[11], kps[13], kps[7]
        else:
            side = "Right"
            # Shoulder=6, Hip=12, Knee=14, Elbow=8
            return side, kps[6], kps[12], kps[14], kps[8]

    def analyze_frame(self, kps, frame_idx):
        self.frame_count = frame_idx
        
        # 1. Smooth
        kps = self.smooth_keypoints(kps)
        
        # Get Keypoints
        side, s, h, k, e = self.get_side_points(kps)
        
        # Visibility checks (conf > 0.2 approx)
        s_vis = s[2] > 0.2
        h_vis = h[2] > 0.2
        k_vis = k[2] > 0.2
        
        current_errors = []
        
        if not (s_vis and h_vis):
            return self.state, ["Keypoints missing"]

        # Calculate main angle (S-H-K)
        shk_angle = 0
        
        if s_vis and h_vis and k_vis:
            shk_angle = calculate_angle(s, h, k)
        elif s_vis and h_vis and self.history_shk_angles:
             # Keep last value or 0? 0 for now.
             pass
 
        # Decide which to use for logic (user pref or auto)
        # Prioritize knee (shk)
        angle = shk_angle 
        if not k_vis:
             angle = 0 # Cannot determine

        sh_dist = calculate_distance(s, h)
        hip_height = h[1] # y coordinate
        
        self.history_shk_angles.append((frame_idx, shk_angle))
        self.history_sh_dist.append((frame_idx, sh_dist))
        self.history_hip_traj.append((frame_idx, h[0], h[1]))

        # --- State Machine ---
        
        # Start Condition (Standing)
        is_straight = (angle > 165) # Relaxed threshold slightly
        
        if self.state == STATE_IDLE:
            if is_straight:
                # Calibrate Start
                self.start_height = hip_height
                self.start_sh_dist = sh_dist
                # Ready for rep
            elif angle < 165 and self.start_sh_dist is not None:
                # Started descending
                self.state = STATE_ECCENTRIC
                self.eccentric_start_time = time.time()
                self.errors = [] # Clear errors for new rep
        
        elif self.state == STATE_ECCENTRIC:
            # Transition to Concentric if angle starts increasing
            if angle > 165:
                pass
                
            # Distance Check: "Back is bending"
            # Distance Check: "Back is bending"
            if sh_dist < 0.8 * self.start_sh_dist:
                self.log_error("Back is bending")
            
            # Detect turning point
            # If angle increases by margin, switch to concentric
            if len(self.history_shk_angles) > 5:
                recent_angles = [x[1] for x in self.history_shk_angles[-5:] if x[1] > 0]
                
                if recent_angles and angle > min(recent_angles) + 5: # Rising
                     self.state = STATE_CONCENTRIC
                     self.concentric_start_time = time.time()
                     
                     # Check Eccentric speed
                     duration = self.concentric_start_time - self.eccentric_start_time
                     duration = self.concentric_start_time - self.eccentric_start_time
                     
                     # Store Eccentric Time
                     curr_rep = self.reps + 1
                     if curr_rep not in self.rep_timings: self.rep_timings[curr_rep] = {}
                     self.rep_timings[curr_rep]['des'] = duration

                     # Check Eccentric speed
                     if duration < 1.0:
                         self.log_error("Uncontrolled eccentric")
        
        elif self.state == STATE_CONCENTRIC:
            # Monitor Fatigue
            duration = time.time() - self.concentric_start_time
            if duration > 5.0:
                self.log_error("Mechanical failure")
            elif duration > 3.0:
                self.log_error("Mechanical fatigue")
            
            # Back check
            # Back check
            if sh_dist < 0.8 * self.start_sh_dist:
                 self.log_error("Back is bending")

            # Check Lockout
            if is_straight:
                # Capture Concentric Time
                duration = time.time() - self.concentric_start_time
                curr_rep = self.reps + 1
                if curr_rep not in self.rep_timings: self.rep_timings[curr_rep] = {}
                self.rep_timings[curr_rep]['asc'] = duration
                
                self.state = STATE_LOCKOUT
                self.reps += 1
                # Mark rep complete
        
        if self.state == STATE_LOCKOUT:
            # Debounce
            if not is_straight:
                self.state = STATE_ECCENTRIC
                self.eccentric_start_time = time.time()
                self.errors = []

        # Hip Trajectory Check (Horizontal deviation)
        if self.start_height is not None:
             # Estimated thigh length
             leg_len = calculate_distance(h, k) * 2
             if leg_len > 0:
                 drop = (h[1] - self.start_height) # Y increases downwards in image?
                 # If Y increases (goes down)
                 if drop > 0.2 * leg_len:
                     self.log_error("Hip hinge movement is incorrect")


        return self.state, self.errors

# --- Main App ---
def main():
    parser = argparse.ArgumentParser(description="Deadlift Analyzer")
    parser.add_argument("video_path", nargs="?", help="Path to video file")
    parser.add_argument("--model", help="Path to .hef model file (e.g. yolov8m_pose.hef)")
    args = parser.parse_args()

    video_path = args.video_path
    
    # Model Selection Logic
    if args.model:
        model_path = args.model
    else:
        print("\n--- Model Selection ---")
        print("1. yolov8s-pose (Fast, Default)")
        print("2. yolov8m-pose (More Accurate, Slower)")
        choice = input("Select model [1/2]: ").strip()
        
        if choice == "2":
            model_path = DEFAULT_MODEL_M
        else:
            model_path = DEFAULT_MODEL_S

    # Check local Directory fallback
    if not os.path.exists(model_path):
        # Try finding it in current dir
        local_name = os.path.basename(model_path)
        if os.path.exists(local_name):
            model_path = local_name
        elif os.path.exists(os.path.join(current_dir, local_name)):
            model_path = os.path.join(current_dir, local_name)
    
    if not video_path:
        # Find recent videos
        print("\n--- Scanning for recent videos ---")
        # Find all .h264 files recursively
        files = glob.glob("**/*.h264", recursive=True)
        # Exclude those that look like output, though output is mp4 usually.
        # record.py outputs h264.
        
        # Sort by modification time
        files = sorted(files, key=os.path.getmtime, reverse=True)
        files = files[:10] # Top 10
        
        if not files:
            print("No recent videos found.")
            video_path = input("Enter path to recorded video: ").strip()
        else:
            print("\nRecent Videos:")
            for i, f in enumerate(files):
                print(f"{i+1}. {f}")
            print(f"{len(files)+1}. Enter custom path")
            
            choice = input(f"Select video [1-{len(files)+1}]: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    video_path = files[idx]
                else:
                    video_path = input("Enter path to recorded video: ").strip()
            else:
                 video_path = input("Enter path to recorded video: ").strip()
        
    if not video_path or not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please ensure you have downloaded the model (e.g. yolov8m_pose.hef).")
        sys.exit(1)

    print(f"Initializing Hailo inference with model: {model_path}")
    
    # 1. Init Inference
    try:
        hailo_infer = HailoInfer(model_path, batch_size=1, output_type="FLOAT32")
    except Exception as e:
        print(f"Failed to init Hailo: {e}")
        sys.exit(1)
        
    # 2. Init Post-Process
    post_proc = PoseEstPostProcessing(
        max_detections=10, # Increase to detect multiple people for tracking
        score_threshold=CONF_THRESHOLD,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )
    
    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 0: total_frames = 0
    
    # Output Video
    # Extract model name for suffix
    model_name = os.path.basename(model_path).replace(".hef", "").replace("_pose", "")
    suffix = f"_analyzed_{model_name}.mp4"
    
    out_path = video_path.replace(".h264", "").replace(".mp4", "") + suffix
    if out_path == video_path: out_path += "_out.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or avc1
    # Use Async Writer
    out_writer = AsyncVideoWriter(out_path, fourcc, fps, (width, height))
    
    analyzer = DeadliftAnalyzer()
    
    model_h, model_w, _ = hailo_infer.get_input_shape()
    
    if total_frames > 0:
        print(f"Processing {total_frames} frames...")
        pbar = tqdm(total=total_frames, unit="frames")
    else:
        print("Processing frames (count unknown)...")
        pbar = tqdm(unit="frames")
    
    frame_idx = 0
    
    # Actually, async is faster. Let's use the callback pattern.
    
    track_box = None # Tracking state [ymin, xmin, ymax, xmax]

    def callback(completion_info, bindings_list, input_batch, output_queue):
        if completion_info.exception:
            print(f"Error: {completion_info.exception}")
            return
        # Get result
        for i, bindings in enumerate(bindings_list):
            # Assuming 1 output or dict
            # pose model usually has multiple outputs.
            # The example uses bindings.output(name).get_buffer()
            # We need to reconstruct the dict expected by post_process
            # The PostProc expects 'raw_detections' dict.
            res = {}
            for name in bindings._output_names:
                res[name] = np.expand_dims(bindings.output(name).get_buffer(), axis=0)
            output_queue.put((input_batch[i], res))

    # Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Preprocess
        # Resize/Letterbox to model_w, model_h
        # But hailo_apps.preprocess handles this? 
        # We'll do simple resize or use provided utils if available.
        # Let's do manual letterbox to be safe and dependent-free.
        
        scale = min(model_w / width, model_h / height)
        nw, nh = int(width * scale), int(height * scale)
        resized = cv2.resize(frame, (nw, nh))
        
        # Pad
        canvas = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
        dx = (model_w - nw) // 2
        dy = (model_h - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw, :] = resized
        
        # Infer
        # We need a list of frames [canvas]
        # HailoInfer.run expects [frames], callback
        
        # Use a list to store result from callback (closure)
        res_container = []
        def sync_cb(completion_info, bindings_list):
             if completion_info.exception: return
             for bindings in bindings_list:
                 res = {}
                 for name in bindings._output_names:
                     res[name] = np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                 res_container.append(res)

        hailo_infer.run([canvas], sync_cb)
        # Wait? run is async.
        # HailoInfer doesn't have wait().
        # We must use a queue or Semaphore.
        
        # Actually, let's use the simple flow: run -> wait -> process.
        # But run() returns immediately.
        # We'll assume for single batch, we can define a status flag.
        
        # Better: Put into queue and wait.
        
        # WAIT: The example uses a queue.
        # I'll stick to synchronous logic by sleeping/waiting on queue.
        
        # ... (Inference Code) ...
        # Since I can't implement complex async here easily without testing,
        # I'll assume I can wait for the callback.
        
        timeout = 2.0
        start_wait = time.time()
        while not res_container and (time.time() - start_wait < timeout):
            time.sleep(0.001)
            
        if not res_container:
            print("Timeout waiting for inference")
            # Write original frame to avoid skip
            out_writer.write(frame)
            continue
            
        raw_detections = res_container[0]
        
        # Post Process
        # class_num=1 for Person
        predictions = post_proc.post_process(raw_detections, model_h, model_w, class_num=1)
        
        # Predictions format: {'predictions': [bboxes, scores, kps, joint_scores]} or similar
        # Utils code returns a DICT with 'bboxes', 'keypoints' etc.
        
        # Filter closest person
        bboxes = predictions['bboxes'][0] # batch 0
        scores = predictions['scores'][0]
        kps = predictions['keypoints'][0]
        joint_scores = predictions['joint_scores'][0]
        num_dets = int(predictions['num_detections'][0]) if 'num_detections' in predictions else len(bboxes)
        # Note: num_detections might be inside the dict structure differently check utils source.
        # Step 29: output['bboxes'][b, :nms_res...
        # So it's padded. We need to find valid ones.
        
        # Find best person (Tracking)
        best_idx = -1
        
        # Parse all valid detections first
        valid_candidates = []
        for i in range(len(scores)):
            score = scores[i]
            if score > CONF_THRESHOLD:
                # bboxes is [ymin, xmin, ymax, xmax] (likely normalized or model coords)
                # Ensure we pass the raw box for tracking logic if standardized
                box = bboxes[i] 
                ymin, xmin, ymax, xmax = box
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                valid_candidates.append({'idx': i, 'score': score, 'cx': cx, 'cy': cy, 'box': box})
        
        if valid_candidates:
            if track_box is None:
                # First time: Prefer Center + Score
                image_cx = model_w / 2
                image_cy = model_h / 2
                
                # Sort by distance to image center
                for cand in valid_candidates:
                    dist_to_center = (cand['cx'] - image_cx)**2 + (cand['cy'] - image_cy)**2
                    cand['center_dist'] = dist_to_center
                
                # Sort by center distance
                valid_candidates.sort(key=lambda x: x['center_dist'])
                
                best_cand = valid_candidates[0]
                best_idx = best_cand['idx']
                track_box = best_cand['box']
            else:
                # Tracking: IoU Overlap
                # Calculate IoU with track_box
                for cand in valid_candidates:
                    iou = calculate_iou(cand['box'], track_box)
                    cand['iou'] = iou
                
                # Filter by Threshold (e.g. 0.2 overlap)
                matched = [c for c in valid_candidates if c['iou'] > 0.2]
                
                if matched:
                    # Sort by IoU desc
                    matched.sort(key=lambda x: x['iou'], reverse=True)
                    best_cand = matched[0]
                    best_idx = best_cand['idx']
                    track_box = best_cand['box'] # Update tracking box
                else:
                    # No overlap with last known box.
                    # Assumption: Lifter is occluded or missing.
                    # Do NOT update track_box. Keep old one.
                    # Do NOT select any candidate (skip analysis for this frame)
                    best_idx = -1 # Explicitly skip
        
        if best_idx >= 0:
            # Map back to original image
            # The PostProc has map_keypoints_to_original_coords function!
            # We can use it.
            
            # Map Box
            box = bboxes[best_idx]
            mapped_box = post_proc.map_box_to_original_coords(box, width, height, model_w, model_h)
            
            # Map KPs
            kp = kps[best_idx] #(17, 2)
            kp_score = joint_scores[best_idx] #(17, 1)
            
            # Combine kp and score
            kp_combined = np.column_stack((kp, kp_score)) #(17, 3)
            
            mapped_kp = post_proc.map_keypoints_to_original_coords(kp_combined[:,:2], width, height, model_w, model_h)
            
            final_kps = np.column_stack((mapped_kp, kp_combined[:,2]))
            
            # Helper to draw
            # ...
            
            # Analyze
            state, errors = analyzer.analyze_frame(final_kps, frame_idx)
            
            # Draw
            # Skeleton
            # Draw
            # Skeleton - Only draw the 6 keypoints for the active side
            # s, h, k, a, w, e are the points.
            # We need them again or just access from final_kps with indices?
            # analyze_frame returned strict subsets logic but main loop draws ALL lines.
            
            # Let's get the side again to know what to draw
            side, s, h, k, e = analyzer.get_side_points(final_kps)
            
            # Connections: S-E, S-H, H-K
            # Points: S, H, K, E
            points_to_draw = [s, h, k, e]
            
            # Draw Points
            for p in points_to_draw:
                if p[2] > 0.2:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)

            # Draw Lines
            # S->E
            if s[2] > 0.2 and e[2] > 0.2:
                 cv2.line(frame, (int(s[0]), int(s[1])), (int(e[0]), int(e[1])), (0, 255, 0), 2)
            # S->H (Skip E->W)
            if s[2] > 0.2 and h[2] > 0.2:
                 cv2.line(frame, (int(s[0]), int(s[1])), (int(h[0]), int(h[1])), (0, 255, 0), 2)
            # H->K
            if h[2] > 0.2 and k[2] > 0.2:
                 cv2.line(frame, (int(h[0]), int(h[1])), (int(k[0]), int(k[1])), (0, 255, 0), 2)
            # Link K->A Removed
                    
            # Text
            cv2.putText(frame, f"Reps: {analyzer.reps}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Rep-by-Rep Analysis Display
            # Top-right corner, listing reps
            x_start = width - 400
            y_start = 50
            line_height = 30
            
            # Determine range of reps to show
            # ONLY SHOW COMPLETED REPS (1 to analyzer.reps)
            # The current rep (analyzer.reps + 1) is still in progress, so don't show it yet.
            max_r = analyzer.reps
            
            y_curr = y_start
            
            for r in range(1, max_r + 1):
                errors = analyzer.rep_history.get(r, [])
                timings = analyzer.rep_timings.get(r, {})
                t_asc = timings.get('asc', 0.0)
                t_des = timings.get('des', 0.0)
                
                # Format: Rep X (tAsc: 1.2s, tDes: 0.9s):
                header_text = f"Rep {r}:"
                if t_asc > 0 or t_des > 0:
                    header_text += f" (tAsc:{t_asc:.1f}s tDes:{t_des:.1f}s)"
                
                if not errors:
                     # Good
                     header_text += " Good"
                     cv2.putText(frame, header_text, (x_start, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                     y_curr += line_height
                else:
                     # Bad
                     cv2.putText(frame, header_text, (x_start, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                     y_curr += line_height
                     for err in errors:
                         cv2.putText(frame, f"- {err}", (x_start + 20, y_curr), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                         y_curr += line_height
                
                y_curr += 10 # Spacing between reps
            
            curr_angle = 0
            if analyzer.history_shk_angles and analyzer.history_shk_angles[-1][1] > 0:
                 curr_angle = analyzer.history_shk_angles[-1][1]
            
            cv2.putText(frame, f"Angle: {int(curr_angle)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        out_writer.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out_writer.release()
    hailo_infer.close()
    
    # Graphs
    # Graphs
    if analyzer.history_shk_angles:
        times = [x[0] for x in analyzer.history_shk_angles]
        shk = [x[1] for x in analyzer.history_shk_angles]
        dists = [x[1] for x in analyzer.history_sh_dist]
        
        valid_traj = [t for t in analyzer.history_hip_traj if t[1] > 0 and t[2] > 0]
        traj_x = [t[1] for t in valid_traj]
        traj_y = [t[2] for t in valid_traj]
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. SHK Angle
        axs[0, 0].plot(times, shk, 'g')
        axs[0, 0].set_title("Shoulder-Hip-Knee Angle")
        axs[0, 0].set_xlabel("Frame")
        axs[0, 0].set_ylabel("Deg")
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. SHK Angle
        axs[0, 0].plot(times, shk, 'g')
        axs[0, 0].set_title("Shoulder-Hip-Knee Angle")
        axs[0, 0].set_xlabel("Frame")
        axs[0, 0].set_ylabel("Deg")
        
        # 2. Ascending/Descending Times (Bar Chart)
        reps = sorted(analyzer.rep_timings.keys())
        asc_times = [analyzer.rep_timings[r].get('asc', 0) for r in reps]
        des_times = [analyzer.rep_timings[r].get('des', 0) for r in reps]
        
        x = np.arange(len(reps))
        width = 0.35
        
        if reps:
            rects1 = axs[0, 1].bar(x - width/2, asc_times, width, label='Ascent')
            rects2 = axs[0, 1].bar(x + width/2, des_times, width, label='Descent')
            axs[0, 1].set_ylabel('Seconds')
            axs[0, 1].set_title('Rep Timings')
            axs[0, 1].set_xticks(x)
            axs[0, 1].set_xticklabels([f"Rep {r}" for r in reps])
            axs[0, 1].legend()
        else:
            axs[0, 1].text(0.5, 0.5, "No Reps Recorded", ha='center')

        # 3. SH Distance
        axs[1, 0].plot(times, dists, 'r')
        axs[1, 0].set_title("Hip-Shoulder Distance (Back Bending)")
        axs[1, 0].set_xlabel("Frame")
        axs[1, 0].set_ylabel("Pixels")
        
        # 4. Rep Summary Table
        axs[1, 1].axis('off')
        
        # Prepare table data
        table_data = []
        # Columns: Rep, Status, Time
        # Status = "Good" or list of errors
        
        for r in reps:
            errs = analyzer.rep_history.get(r, [])
            status = "Good" if not errs else "\n".join(errs)
            t_asc = analyzer.rep_timings[r].get('asc', 0)
            t_des = analyzer.rep_timings[r].get('des', 0)
            time_str = f"Up: {t_asc:.1f}s\nDown: {t_des:.1f}s"
            table_data.append([f"Rep {r}", status, time_str])
            
        if table_data:
            table = axs[1, 1].table(cellText=table_data, colLabels=["Rep", "Analysis", "Timings"], loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            axs[1, 1].set_title("Rep Analysis Summary")
        else:
             axs[1, 1].text(0.5, 0.5, "No Analysis Data", ha='center')
        
        plt.tight_layout()
        plt.savefig(video_path + f"_analysis_graphs_{model_name}.png")
        print("Detailed graphs saved.")

if __name__ == "__main__":
    main()
