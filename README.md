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
    Download `yolov8s_pose.hef` or `yolov8n_pose.hef` from the Hailo Model Zoo and place them in the project folder.

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

### 3. Pipelined Inference (`AsyncVideoWriter`)
To achieve high FPS, we decoupled the **Inference Loop** from the **Video Writing** (I/O) operation. The main thread pushes processed frames to a queue, and a background thread handles the slow `.mp4` encoding.

```python
class AsyncVideoWriter:
    def _run(self):
        while self.running:
            # Write frames from buffer without blocking the AI model
            frame = self.queue.get()
            self.writer.write(frame)
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

1.  **Pipelining (Threading)**: 
    *   *Problem*: `cv2.VideoWriter` is CPU-heavy and blocks execution.
    *   *Solution*: Using `AsyncVideoWriter` allows the Hailo NPU to process the *next* frame while the CPU writes the *previous* frame to disk simultaneously.

2.  **Model Selection**:
    *   **YOLOv8-Nano (`n`)**: Extremely fast (~40-50 FPS), good for general flow.
    *   **YOLOv8-Small (`s`)**: Balanced accuracy (~30 FPS).
    *   **YOLOv8-Medium (`m`)**: High precision, slower (<15 FPS).

## 📖 Usage Guide

### Main Menu
Run `python menu.py` to access the unified interface.
- **Option 1 (Record)**: Uses `Picamera2` to capture raw `.h264` footage. Raw capture prevents dropped frames during recording.
- **Option 2 (Analyze)**: Selects the AI model and video file.
- **Option 3 (Auto)**: Chains both steps for a seamless "Record -> Evaluate" workout flow.

## 📂 Git & Version Control

This repository is configured to ignore heavy media files by default.
- **Included**: Source code (`.py`), Documentation (`.md`).
- **Excluded**: `*.h264`, `*.mp4`, `*.hef` (Models), and session folders.
