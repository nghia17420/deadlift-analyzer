# AI-Powered Deadlift Analyzer 🏋️‍♂️🤖

A real-time, computer vision-based application designed to analyze deadlift form, track repetitions, and provide feedback on performance metrics using a **Raspberry Pi 5** and the **Hailo-8L AI Accelerator**.

This project utilizes the `YOLOv8-Pose` model to detect keypoints and evaluate lifting mechanics such as:
- **Back Bending**: Monitors hip-shoulder distance to detect spinal rounding.
- **Hip Hinge**: Analysis of hip trajectory and vertical deviation.
- **Timing Analysis**: Tracks eccentric (lowering) and concentric (lifting) phase durations.
- **Rep Tracking**: Validates reps based on lockout angles and movement thresholds.

## 🚀 Features

- **Real-time Pose Estimation**: High-performance inference (30+ FPS) using Hailo-8L NPU.
- **Rep-by-Rep Feedback**: Displays specific errors (e.g., "Back is bending", "Mechanical fatigue") for each completed rep.
- **Timing Metrics**: Records `tAsc` (Time to Ascend) and `tDes` (Time to Descend) for every rep.
- **Automated Workflow**: 
  - `record.py`: Captures high-quality video efficiently.
  - `analyze.py`: Processes videos with AI pipeline.
  - `menu.py`: Unified interface for recording and analysis.
- **Performance Graphs**: Generates post-workout visualization of joint angles, hip trajectory, and rep timings.
- **Pipelined Architecture**: Uses multi-threading to separate AI inference from video writing for maximum speed.

## 🛠 Hardware Requirements

- **Raspberry Pi 5** (8GB RAM recommended)
- **Hailo-8L AI Kit** (M.2 Key M module + Hat)
- **Raspberry Pi Camera Module 3** (or compatible Picamera2 device)
- **Active Cooling** (Recommended for sustained inference)

## 📦 Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/deadlift-analyzer.git
    cd deadlift-analyzer
    ```

2.  **Install Dependencies**:
    Ensure you have the Hailo TAPPAS and HailoRT installed on your Raspberry Pi 5.
    Then install Python requirements:
    ```bash
    pip install opencv-python numpy matplotlib tqdm
    ```
    *Note: `hailo_apps` and `picamera2` are typically pre-installed on the official Raspberry Pi OS (Bookworm) with Hailo software.*

3.  **Model Setup**:
    This project requires compiled HEF (Hailo Executable Format) models.
    - Download `yolov8s_pose.hef` (Small) or `yolov8n_pose.hef` (Nano) from the Hailo Model Zoo.
    - Place them in `/usr/local/hailo/resources/models/hailo8l/` OR in the project root directory.

## 📖 Usage

### Main Menu
Run the unified menu to access all features:
```bash
python menu.py
```
You will be presented with three options:
1. **Record Only**: Capture a new lifting session.
2. **Analyze Only**: Process an existing `.h264` video file.
3. **Record & Analyze**: Automatically record and then immediately process the video.

### Recording (`record.py`)
- Captures raw `.h264` video for low latency and high quality.
- Saves videos in subfolders named `[LifterName]_[Weight]_[Date]`.

### Analyzing (`analyze.py`)
- Performs pose estimation on the video.
- **Interactive**: Allows selecting between Speed (Nano/Small models) vs Accuracy (Medium model).
- **Output**: 
  - Generates an overlay video `*_analyzed.mp4` with skeleton tracking and feedback.
  - Saves a `_analysis_graphs.png` with performance charts.

## 📊 Thresholds & Logic

| Metric | Threshold | logic |
| :--- | :--- | :--- |
| **Lockout Angle** | **> 165°** | Angle (Shoulder-Hip-Knee) for valid rep start/end. |
| **Back Bending** | **< 80%** | Flagged if Shoulder-Hip distance compresses significantly. |
| **Hip Hinge** | **> 20%** | Flagged if hips drop vertically (squatting instead of hinging). |
| **Eccentric Fail** | **< 1.0s** | "Uncontrolled eccentric" (dropping the weight too fast). |
| **Fatigue Warn** | **> 3.0s** | "Mechanical fatigue" (slow ascent). |
| **Failure** | **> 5.0s** | "Mechanical failure" (grinding/stuck). |

## 📂 Project Structure

- `menu.py`: Entry point CLI.
- `record.py`: Camera handling and video validation.
- `analyze.py`: Core logic, AI inference, State Machine, and Visualization.
- `record_and_analyze.py`: Wrapper for automation.
- `hailo-apps/`: (Optional) Local link to Hailo app utilities if not in global path.

## 🛑 Troubleshooting

- **"Processing 0 frames"**: Usually happens with raw `.h264` files as they lack header metadata. The analyzer still processes them correctly.
- **Low FPS**: 
  - Ensure you are using the `AsyncVideoWriter` pipeline (enabled by default).
  - Switch to `yolov8n_pose.hef` for faster inference.
  - Disable video preview in recording if CPU is throttled.

---
*Created by RaspBenny & The Google DeepMind Team*
