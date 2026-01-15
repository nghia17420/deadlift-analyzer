#!/usr/bin/env python3
import os
import subprocess
import argparse
import record

def main():
    parser = argparse.ArgumentParser(description="Record and Analyze Deadlift")
    parser.add_argument("--no-preview", action="store_true", help="Disable video preview window")
    args = parser.parse_args()
    
    # Interactive Menu if no args provided (default behavior usually needs to decide)
    # But since we want "same options", let's mimic the record.py menu exactly
    
    print("\n--- Record & Analyze Deadlift ---")
    print("1. Record with Preview (Close window to stop)")
    print("2. Record without Preview (Ctrl+C to stop)")
    
    # If user passed --no-preview, we can default to that, or just ignore args and always ask?
    # The user request says "running it will display options".
    # So we should probably prioritize the menu, or use args if present to skip menu (headless mode).
    # Let's use the menu by default for better UX.
    
    if args.no_preview:
        use_preview = False
        print("Option selected via argument: No Preview")
    else:
        choice = input("Select option [1/2]: ").strip()
        use_preview = True # Default
        if choice == "2":
            use_preview = False

    # 1. Record
    filepath = record.record_video(preview=use_preview)
    
    if not filepath or not os.path.exists(filepath):
        print("Recording failed or file not found.")
        return

    print("\n" + "="*40)
    print("Recording Finished. Starting Analysis...")
    print("="*40 + "\n")
    
    import sys
    # 2. Analyze
    try:
        analyze_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), "analyze.py")
        # Pass the filepath to the script
        subprocess.run([sys.executable, analyze_script, filepath], check=True)
    except Exception as e:
        print(f"Error running analysis: {e}")

if __name__ == "__main__":
    main()
