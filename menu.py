#!/usr/bin/env python3
import sys
import os
import subprocess

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    while True:
        clear_screen()
        print("="*40)
        print("   DEADLIFT ANALYZER - MAIN MENU")
        print("="*40)
        print("1. Record Only")
        print("   (Capture video, save to folder)")
        print("")
        print("2. Analyze Only")
        print("   (Process existing video, select model)")
        print("")
        print("3. Record & Analyze")
        print("   (Chain steps)")
        print("")
        print("q. Quit")
        print("="*40)
        
        choice = input("Select mode: ").strip().lower()
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        
        if choice == '1':
            try:
                subprocess.run([sys.executable, os.path.join(script_dir, "record.py")])
            except KeyboardInterrupt:
                pass
            input("\nPress Enter to return to menu...")
            
        elif choice == '2':
            try:
                subprocess.run([sys.executable, os.path.join(script_dir, "analyze.py")])
            except KeyboardInterrupt:
                pass
            input("\nPress Enter to return to menu...")
            
        elif choice == '3':
            try:
                subprocess.run([sys.executable, os.path.join(script_dir, "record_and_analyze.py")])
            except KeyboardInterrupt:
                pass
            input("\nPress Enter to return to menu...")
            
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            input("Invalid selection. Press Enter to try again...")

if __name__ == "__main__":
    main()
