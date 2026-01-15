#!/usr/bin/env python3
import sys
import os
import time
import argparse
import numpy as np
import cv2
import queue
import threading
from datetime import datetime
from picamera2 import Picamera2

# Add current directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import necessary components from analyze.py
try:
    from analyze import (
        DeadliftAnalyzer,
        HailoInfer,
        PoseEstPostProcessing,
        calculate_iou,
        AsyncVideoWriter,
        DEFAULT_MODEL_S,
        CONF_THRESHOLD
    )
except ImportError as e:
    print(f"Error importing from analyze.py: {e}")
    sys.exit(1)

def make_filename(lifter_name: str, weight_kg: str) -> str:
    # Normalise input strings
    name_clean = lifter_name.strip().replace(" ", "_").lower()
    weight_clean = weight_kg.strip().replace(" ", "")
    # Short date-time format: YYYY-MM-DD_HH-MM
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Format: lifter_weight_YYYY-MM-DD_HH-MM.mp4
    return f"{name_clean}_{weight_clean}_{stamp}.mp4"

def main():
    parser = argparse.ArgumentParser(description="Live Deadlift Analyzer")
    # Removed --model argument as we are forcing S model or auto-detecting
    parser.add_argument("--width", type=int, default=1280, help="Preview width")
    parser.add_argument("--height", type=int, default=720, help="Preview height")
    args = parser.parse_args()

    print("\n--- Live Deadlift Analysis ---")
    lifter_name = input("Lifter name: ")
    weight_kg = input("Lifting weight (e.g. 80kg): ")

    filename = make_filename(lifter_name, weight_kg)
    session_dir = filename.replace(".mp4", "")
    os.makedirs(session_dir, exist_ok=True)
    
    # Save as _analyzed.mp4 to match convention or just .mp4? 
    # User said "save into a folder like other functions". 
    # record.py saves <name>.h264 in <name>/
    # analyze.py produces <name>_analyzed.mp4
    # Since this is ALREADY analyzed, let's call it <name>_analyzed.mp4 just to be clear, 
    # or just <name>.mp4 implies it's the result. 
    # Let's stick to the record.py naming for the FOLDER, but the file will be .mp4
    
    out_path = os.path.join(session_dir, filename)
    print(f"Output file will be saved to: {out_path}")

    # FORCE Model S
    model_path = DEFAULT_MODEL_S

    # Locate Model
    if not os.path.exists(model_path):
        local_name = os.path.basename(model_path)
        if os.path.exists(local_name):
            model_path = local_name
        elif os.path.exists(os.path.join(current_dir, local_name)):
            model_path = os.path.join(current_dir, local_name)
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Ensure yolov8s_pose.hef is present.")
        sys.exit(1)

    print(f"Initializing Hailo inference with model: {model_path}")

    # Init Hailo
    try:
        hailo_infer = HailoInfer(model_path, batch_size=1, output_type="FLOAT32")
    except Exception as e:
        print(f"Failed to init Hailo: {e}")
        sys.exit(1)

    # Init PostProcessing
    post_proc = PoseEstPostProcessing(
        max_detections=10, 
        score_threshold=CONF_THRESHOLD,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )

    model_h, model_w, _ = hailo_infer.get_input_shape()
    # print(f"Model Input Shape: {model_w}x{model_h}")

    # Init Analyzer
    analyzer = DeadliftAnalyzer()

    # Picamera2 Setup
    picam2 = Picamera2()
    
    # Video Configuration
    # We want 30 FPS.
    fps = 30
    config = picam2.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}, 
        buffer_count=6,
        controls={"FrameDurationLimits": (33333, 33333)}
    )
    picam2.configure(config)
    picam2.start()

    # Init Video Writer
    # Use AsyncVideoWriter from analyze.py for performance
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = AsyncVideoWriter(out_path, fourcc, fps, (args.width, args.height))

    print("Starting Live Inference & Recording... Press 'q' or Ctrl+C to stop.")

    track_box = None
    frame_idx = 0

    try:
        while True:
            # Capture Frame
            frame = picam2.capture_array("main")
            if frame is None:
                continue

            frame_idx += 1
            h, w = frame.shape[:2]

            # Preprocess
            scale = min(model_w / w, model_h / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh))

            canvas = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
            dx = (model_w - nw) // 2
            dy = (model_h - nh) // 2
            canvas[dy:dy+nh, dx:dx+nw, :] = resized

            # Inference
            raw_detections = None
            def sync_cb(completion_info, bindings_list):
                 nonlocal raw_detections
                 if completion_info.exception: return
                 for bindings in bindings_list:
                     res = {}
                     for name in bindings._output_names:
                         res[name] = np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                     raw_detections = res

            hailo_infer.run([canvas], sync_cb)
            
            # Polling wait
            start_wait = time.time()
            while raw_detections is None and (time.time() - start_wait < 0.2):
                time.sleep(0.001)

            if raw_detections:
                # Post Process
                predictions = post_proc.post_process(raw_detections, model_h, model_w, class_num=1)
                
                bboxes = predictions['bboxes'][0]
                scores = predictions['scores'][0]
                kps = predictions['keypoints'][0]
                joint_scores = predictions['joint_scores'][0]
                
                # Tracking Logic
                best_idx = -1
                valid_candidates = []
                for i in range(len(scores)):
                    score = scores[i]
                    if score > CONF_THRESHOLD:
                        box = bboxes[i]
                        valid_candidates.append({'idx': i, 'score': score, 'box': box})

                if valid_candidates:
                    if track_box is None:
                        # Center closest
                        image_cx = model_w / 2
                        image_cy = model_h / 2
                        for c in valid_candidates:
                             ymin, xmin, ymax, xmax = c['box']
                             cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
                             c['dist'] = (cx-image_cx)**2 + (cy-image_cy)**2
                             
                        valid_candidates.sort(key=lambda x: x['dist'])
                        best_cand = valid_candidates[0]
                        best_idx = best_cand['idx']
                        track_box = best_cand['box']
                    else:
                        # IoU Tracking
                        for cand in valid_candidates:
                            cand['iou'] = calculate_iou(cand['box'], track_box)
                        matched = [c for c in valid_candidates if c.get('iou', 0) > 0.2]
                        if matched:
                            matched.sort(key=lambda x: x['iou'], reverse=True)
                            best_cand = matched[0]
                            best_idx = best_cand['idx']
                            track_box = best_cand['box']
                
                if best_idx >= 0:
                     # Map KPs
                     kp = kps[best_idx]
                     kp_score = joint_scores[best_idx]
                     kp_combined = np.column_stack((kp, kp_score))
                     
                     mapped_kp = post_proc.map_keypoints_to_original_coords(kp_combined[:,:2], w, h, model_w, model_h)
                     final_kps = np.column_stack((mapped_kp, kp_combined[:,2]))
                     
                     # Analyze
                     state, errors = analyzer.analyze_frame(final_kps, frame_idx)
                     
                     # Draw
                     side, s, h, k, e = analyzer.get_side_points(final_kps)
                     points_to_draw = [s, h, k, e]
                     for p in points_to_draw:
                        if p[2] > 0.2:
                            cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
                     
                     if s[2]>0.2 and h[2]>0.2: cv2.line(frame, (int(s[0]), int(s[1])), (int(h[0]), int(h[1])), (0, 255, 0), 2)
                     if h[2]>0.2 and k[2]>0.2: cv2.line(frame, (int(h[0]), int(h[1])), (int(k[0]), int(k[1])), (0, 255, 0), 2)
                     if s[2]>0.2 and e[2]>0.2: cv2.line(frame, (int(s[0]), int(s[1])), (int(e[0]), int(e[1])), (0, 255, 0), 2)
                     
                     # Overlay Text
                     cv2.putText(frame, f"State: {state}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                     cv2.putText(frame, f"Reps: {analyzer.reps}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                     
                     if errors:
                         for i, err in enumerate(errors):
                             cv2.putText(frame, f"! {err}", (30, 140 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                     else:
                         cv2.putText(frame, "Good Form", (w-250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Show Frame
            cv2.imshow("Deadlift Analyzer - Live", frame)
            
            # Write Frame
            writer.write(frame)
            
            key = cv2.waitKey(1) & 0xFF
            # Check if 'q' is pressed OR if window is closed (click X)
            if key == ord('q') or cv2.getWindowProperty("Deadlift Analyzer - Live", cv2.WND_PROP_VISIBLE) < 1:
                break
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error in loop: {e}")
    finally:
        picam2.stop()
        picam2.close()
        writer.release()
        cv2.destroyAllWindows()
        hailo_infer.close()
        print(f"Saved recording to: {out_path}")

if __name__ == "__main__":
    main()
