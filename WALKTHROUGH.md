# Deadlift Analyzer Walkthrough

The Deadlift Analyzer has been implemented in `analyze.py`. It uses the Hailo-8L accelerator to run YOLOv8 Pose estimation and analyzes your deadlift form based on geometric constraints.

## Features
- **Side-Specific Analysis**: Automatically detects the dominant side and tracks only 6 keypoints: Shoulder, Hip, Knee, Ankle, Wrist, Elbow.
- **Top-Down Analysis**: Optimised for RDL/Romanian Deadlift style (starts standing).
- **Rep Counting**: Counts reps based on hip/knee extension (Start/Lockout angles > 170-175°).
- **Error Detection**:
  - **Uncontrolled Eccentric**: Descending too fast (< 1s).
  - **Mechanical Fatigue**: Ascending too slow (> 2s).
  - **Mechanical Failure**: Ascending > 3s or stall.
  - **Back Bending**: Torso shortens (Shoulder-Hip distance decreases) by > 10%.
  - **Hip Trajectory**: Hips drop vertically > 20% of leg length (Squatting instead of Hinging).

## How to Use

1.  **Start the App**:
    ```bash
    python3 menu.py
    ```
    Select one of the 3 modes:

    - **1. Record Only**:
        - Shows a menu to choose **Preview** or **No Preview**.
        - Enter `Lifter Name` and `Weight`.
        - **To Stop**: Press `q`, close the window, or Ctrl+C.

    - **2. Analyze Only**:
        - Select existing video (Recent list or Path).
        - Select Model (Small/Medium).

    - **3.## Workflow Automation
    - `menu.py`: Main entry point.
    - `record.py`: Handles video capture.
    - `analyze.py`: Handles inference and visualization.
    - `record_and_analyze.py`: Chains them together.ately starts analysis.

    
    Both scripts will ask for `Lifter Name` and `Weight` and save to a subfolder.

    (Optional) If you want to re-run analysis manually:
    ```bash
    python3 analyze.py <path_to_video>
    ```

3.  **View Results**:
    - **Video**: An output video will be created with the model name:
        - `*_analyzed_yolov8s.mp4` (Small model)
        - `*_analyzed_yolov8m.mp4` (Medium model)
        - Contains: Skeleton overlay, Rep counter, Error List.
        - "Perfect Set" is only displayed if zero errors occurred for the entire session.
        - No skipped frames (original frame is preserved if analysis times out).
    - **Graph**: A composite graph `*_analysis_graphs_yolov8[s/m].png` showing:
        1. Shoulder-Hip-Knee Angle
        2. Shoulder-Hip-Ankle Angle
        3. Hip-Shoulder Distance (pixel units)
        4. Hip Trajectory (Side view X-Y path)

## Verification
- Confirmed model path: `/usr/local/hailo/resources/models/hailo8l/yolov8s_pose.hef`.
- Confirmed dependencies: `hailo_apps` logic is imported directly.
- Implemented state machine for Start -> Bottom -> Lockout.

## Troubleshooting

### Recording: Missing Frames?
If `record.py` produces choppy video or missing frames:
1.  **Buffer Count**: We have now increased the internal buffer to 12 frames to handle SD card write pauses.
2.  **Preview**: If recording is still choppy, try disabling the preview window in `record.py` (comment out `picam2.start_preview`).
3.  **Storage**: Ensure you are recording to a fast SD card or USB 3.0 SSD.

## Next Steps
- Test with a real video.
- Calibrate angle thresholds if needed (currently set to 170-180 range).

## Changing the AI Model
The system uses `yolov8s_pose.hef` by default (Small version).
If you want to use the **Medium (m)** version for better accuracy:
1.  **Selection**:
    When you run `analyze.py` (or when `record.py` auto-runs it), it will ask:
    ```
    Model Selection:
    1. yolov8s-pose (Fast, Default)
    2. yolov8m-pose (More Accurate, Slower)
    ```
    Select `2` to use the medium model.
    (The model has been downloaded and installed for you).

    If you run `python3 analyze.py` without a video file, it will also **scan for recent videos** (top 10 newest) in all subfolders and list them for you to choose.
