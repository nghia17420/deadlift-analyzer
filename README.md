# AI-Powered Deadlift Analyzer 🏋️‍♂️🤖

A real-time, computer vision-based application designed to analyze deadlift form, track repetitions, and provide feedback on performance metrics using a **Raspberry Pi 5** and the **Hailo-8L AI Accelerator**.

This project utilizes the `YOLOv8-Pose` model to detect keypoints and evaluate lifting mechanics. It features a custom state machine for rep logic and a multi-threaded architecture for high-performance inference.

## 🚀 Features & Capabilities

- **Real-time Pose Estimation**: ~30 FPS inference using Hailo-8L NPU.
- **Form Feedback**: Detects specific errors like "Back Bending" or "Incorrect Hip Hinge" during the lift.
- **Rep Timing**: Precision tracking of Ascending (`tAsc`) vs Descending (`tDes`) phases in seconds.
- **Post-Set Analytics**: Generates graphs for Hip Trajectory and Velocity/Timing profiles.

## 🛠 Hardware Requirements

- **Raspberry Pi 5** (8GB RAM recommended)
- **Hailo-8L AI Kit** (M.2 Key M module + Hat)
- **Raspberry Pi Camera Module 3** (or compatible Picamera2 device)
- **Active Cooling** (Highly Recommended)

## 📦 Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/nghia17420/deadlift-analyzer.git
    cd deadlift-analyzer
    ```

2.  **Install Dependencies**:
    Ensure HailoRT/TAPPAS are installed. Then:
    ```bash
    pip install opencv-python numpy matplotlib tqdm
    ```

3.  **Model Setup**:
    Download `yolov8s_pose.hef` (Small) or `yolov8m_pose.hef` (Medium) from the Hailo Model Zoo.
    **Important**: Place the `.hef` file directly in the `deadlift-analyzer/` project folder.

## 📂 Project Structure

Here is an overview of the key files and directories in the project:

-   `analyze.py`: **Core Application**. Handles AI inference, state machine logic, error detection, and video generation.
-   `record.py`: Recording script using `Picamera2` to capture raw `.h264` video.
-   `menu.py`: Interactive CLI menu to easily switch between recording and analyzing.
-   `live_analyze_preview.py`: **Live Assessment**. Runs real-time inference directly from the camera feed, providing immediate visual feedback without saving to disk first.
-   `record_and_analyze.py`: Automation script that records a video and immediately runs analysis on it.
-   `hailo-apps/`: A bundled submodule containing the necessary Hailo NPU interface and post-processing tools.

## 📊 Output Description

After analysis, the tool generates two files in the same directory as the input video:

1.  **Analyzed Video** (e.g., `video_analyzed_yolov8s.mp4`):
    -   **Visual Overlay**: Skeleton overlaid on the lifter.
    -   **Rep Counter**: Tracks completed reps.
    -   **Real-time Feedback**: Displays "Good" or specific error messages on the top right.
    -   **Phase Timings**: Shows Ascent and Descent times for each rep.

2.  **Analysis Graphs** (e.g., `video_analysis_graphs_yolov8s.png`):
    -   **SHK Angle**: Graph of the Shoulder-Hip-Knee angle over time.
    -   **Rep Timings**: Bar chart comparing ascent vs. descent duration for each rep.
    -   **Hip Trajectory**: Plot of hip path stability.
    -   **Summary Table**: Detailed text breakdown of every rep's status.

## ⚙️ Configuration & Thresholds

The analysis logic relies on specific thresholds defined in `analyze.py`. You can adjust these to fit different body types or lifting styles.

**Location to Edit**: Open `analyze.py` and modify the values in the `DeadliftAnalyzer` class or the `analyze_frame` method.

| Metric | Current Threshold | Description | Code Location |
| :--- | :--- | :--- | :--- |
| **Start/Lockout Phase** | `Angle > 165°` | Defines when the lifter is standing straight (hips extended). | `analyze_frame`: `is_straight = (angle > 165)` |
| **Back Bending** | `< 80%` of start dist | Flags if the Hip-Shoulder distance compresses significantly (upper back rounding). | `analyze_frame`: `if sh_dist < 0.8 * self.start_sh_dist` |
| **Hip Hinge Comp.** | `> 20%` of leg length | Alerts if the hips drop too much during the hinge (squatting the weight). | `analyze_frame`: `if drop > 0.2 * leg_len` |
| **Eccentric Speed** | `< 1.0 second` | Warns if the weight is dropped too quickly (Uncontrolled Eccentric). | `analyze_frame`: `if duration < 1.0:` |
| **Fatigue Monitor** | `> 3.0 seconds` | Flags "Mechanical Fatigue" if the concentric phase is slow. | `analyze_frame`: `elif duration > 3.0:` |
| **Side Detection** | Confidence > 0.2 | Auto-detects which side (left/right) is visible based on keypoint confidence. | `get_side_points` method |

## 🧩 Code Structure & Key Functions

The core logic resides in `analyze.py`. Here is a breakdown of the critical components:

### 1. The State Machine (`DeadliftAnalyzer`)
Logic to track the lifter's phase (Eccentric vs Concentric) and validate reps based on joint angles.

```python
def analyze_frame(self, kps, frame_idx):
    # Calculate angle (Shoulder-Hip-Knee)
    angle = calculate_angle(s, h, k)
    
    # State Transitions
    if self.state == STATE_IDLE and angle < 165:
        self.state = STATE_ECCENTRIC
        self.eccentric_start_time = time.time()
        
    elif self.state == STATE_CONCENTRIC and is_straight:
        self.state = STATE_LOCKOUT
        self.reps += 1
        # Capture ascent duration
        duration = time.time() - self.concentric_start_time
        self.rep_timings[self.reps]['asc'] = duration
```

### 2. Form Correction Logic
We continuously monitor vector relationships between keypoints. For example, checking if the back bends under load:

```python
# Monitor Hip-Shoulder distance compression
sh_dist = calculate_distance(shoulder, hip)
if sh_dist < 0.8 * self.start_sh_dist:
    self.log_error("Back is bending")
```

### 3. Threaded Pipeline (`FramePipeline`)
To maximize hardware utilization, we replaced the simple loop with a Producer-Consumer architecture. The main thread continuously feeds the NPU (Hailo) while a background worker processes the results order.

```python
# Producer (Main) -> Async Inference -> Queue -> Consumer (Worker) -> Async Writer
def processing_worker(proc_queue):
    while True:
        # Fetches completed inference results
        frame, detections = proc_queue.get()
        # Post-processes and analyzes without blocking the NPU
        analyze_and_draw(frame, detections)
```

## 📚 Dependencies: Why `hailo-apps`?

We include the [hailo-apps](https://github.com/hailo-ai/hailo-apps) repository in this folder.
- **Reference**: `analyze.py` imports specific modules from it:
  ```python
  from pose_estimation_utils import PoseEstPostProcessing
  from hailo_apps.python.core.common.hailo_inference import HailoInfer
  ```
- **Reason**: The `HailoInfer` class handles the complex low-level communication with the NPU (HEF loading, VDevice config). `PoseEstPostProcessing` efficiently handles the Non-Maximum Suppression (NMS) and decoding of the raw YOLO binaries into usable (x, y) keypoints. By bundling it, we ensure compatibility without forcing a full system-wide installation of the massive TAPPAS suite.

## ⚡ Performance Optimization

We achieved **>30 FPS** generic throughput (video speed) using two main techniques:

1.  **Threaded Inference Pipeline (Producer-Consumer)**: 
    *   *Problem*: Sequential processing (`Read -> Infer -> Process -> Write`) leaves the CPU or NPU idle for significant periods.
    *   *Solution*: We implemented a **multi-stage threaded pipeline**:
        *   **Stage 1 (Main Thread)**: Reads frames and dispatches async inference requests to the NPU.
        *   **Stage 2 (Hailo Async)**: The NPU processes frames in the background.
        *   **Stage 3 (Worker Thread)**: A dedicated consumer thread handles post-processing, tracking, analysis logic, and drawing.
        *   **Stage 4 (Async Writer)**: Video encoding is offloaded to a separate thread.
    *   *Result*: Drastically reduced latency and higher throughput by overlapping all operations.

2.  **Model Selection**:
    *   **YOLOv8-Small (`s`)**: Balanced accuracy (~30+ FPS).
    *   **YOLOv8-Medium (`m`)**: High precision, improved tracking.

3.  **Buffer Control**:
    *   Use the `--buffer` argument to control how many frames can be "in-flight". Higher buffers smooth out jitter but use more RAM.


## 📖 Usage Guide

### Main Menu
Run `python menu.py` to access the unified interface.
- **Option 1 (Record)**: Uses `Picamera2` to capture raw `.h264` video. Raw capture prevents dropped frames during recording.
- **Option 2 (Analyze)**: Selects the AI model and video file.
- **Option 3 (Auto)**: Chains both steps for a seamless "Record -> Evaluate" workout flow.
- **Option 4 (Live Analysis)**: Starts the real-time preview mode. Useful for quick form checks or setup. Automatically saves the output to a timestamped folder.

### Advanced Usage (CLI)
You can run the analyzer directly with additional options:

```bash
python3 analyze.py video.mp4 --model yolov8s_pose.hef --buffer 64
```

- `--buffer [N]`: Sets the size of the internal frame buffer (default: 32). Increase this if you experience dropped frames or jitter, provided you have available RAM.

## 📂 Git & Version Control

This repository is configured to ignore heavy media files by default.
- **Included**: Source code (`.py`), Documentation (`.md`).
- **Excluded**: `*.h264`, `*.mp4`, `*.hef` (Models), and session folders.
