import argparse
import sys
import cv2
import os
import time
import numpy as np
from datetime import datetime

from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FileOutput


def make_filename(lifter_name: str, weight_kg: str) -> str:
    # Normalise input strings
    name_clean = lifter_name.strip().replace(" ", "_")
    weight_clean = weight_kg.strip().replace(" ", "")
    # Short date-time format: YYYY-MM-DD_HH-MM  (no seconds)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Format: lifter_weight_YYYY-MM-DD_HH-MM.h264
    return f"{name_clean}_{weight_clean}_{stamp}.h264"


def record_video(preview=True):
    # Ask user for metadata
    lifter_name = input("Lifter name: ")
    weight_kg = input("Lifting weight (e.g. 80kg): ")

    filename = make_filename(lifter_name, weight_kg)
    
    session_dir = filename.replace(".h264", "")
    os.makedirs(session_dir, exist_ok=True)
    
    filepath = os.path.join(session_dir, filename)
    print(f"Output file: {filepath}")

    picam2 = Picamera2()

    # 1080p @ 30 fps, YUV420 main stream for efficient H.264 encoding
    controls = {"FrameDurationLimits": (33333, 33333)}  # fixed 30 fps
    
    config_args = {
        "main": {"size": (1920, 1080), "format": "YUV420"},
        "controls": controls,
        "buffer_count": 12
    }
    
    if preview:
        config_args["lores"] = {"size": (640, 360), "format": "RGB888"}

    video_config = picam2.create_video_configuration(**config_args)
    picam2.configure(video_config)

    # Medium quality H.264 – good balance for motion and file size
    encoder = H264Encoder()
    encoder.quality = Quality.MEDIUM

    output = FileOutput(filepath)

    picam2.start()

    picam2.start_recording(encoder, output)
    print("Recording workout...")
    
    if preview:
        print("Preview window open. Close window OR Press Ctrl+C to stop.")
    else:
        print("Recording in background. Press Ctrl+C to stop.")

    try:
        if preview:
            while True:
                # Capture frame for preview from lores stream
                frame = picam2.capture_array("lores")
                # Flip if needed (optional, assuming standard orientation)
                # frame = cv2.flip(frame, 1) # Mirror effect?
                
                cv2.imshow("Recording Preview", frame)
                
                # Check for Window Close or 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty("Recording Preview", cv2.WND_PROP_VISIBLE) < 1:
                    print("\nWindow closed.")
                    break
        else:
            # No preview loop
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nStopping recording...")

    picam2.stop_recording()
    picam2.stop()
    picam2.close()
    
    if preview:
        cv2.destroyAllWindows()
    
    print(f"Recording saved to: {filepath}")
    return filepath

def main():
    print("\n--- Record Deadlift Video ---")
    print("1. Record with Preview (Close window to stop)")
    print("2. Record without Preview (Ctrl+C to stop)")
    choice = input("Select option [1/2]: ").strip()
    
    use_preview = True
    if choice == "2":
        use_preview = False
    
    record_video(preview=use_preview)


if __name__ == "__main__":
    main()
