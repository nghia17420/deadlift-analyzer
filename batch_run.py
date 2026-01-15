import os
import subprocess
import glob

models = [
    "/usr/local/hailo/resources/models/hailo8l/yolov8s_pose.hef",
    "/usr/local/hailo/resources/models/hailo8l/yolov8m_pose.hef"
]

files = glob.glob("**/*.h264", recursive=True)
# Filter out files that might be in a different directory or unwanted if needed, 
# but glob recursive from current dir is what we want.
files = sorted(files)

total = len(files) * len(models)
count = 0

print(f"Found {len(files)} files. Total tasks: {total}")

for f in files:
    for m in models:
        count += 1
        print(f"[{count}/{total}] Analyzing {f} with {os.path.basename(m)}...")
        try:
            # Capture output to avoid spamming the main log if desired, or let it through. 
            # Letting it through allows us to see progress in the command output.
            subprocess.run(["python3", "analyze.py", f, "--model", m], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {f} with {m}: {e}")
        except Exception as e:
            print(f"Unexpected error on {f}: {e}")

print("DONE")
