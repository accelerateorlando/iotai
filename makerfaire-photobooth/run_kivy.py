#!/usr/bin/env python3
"""Launcher script for the Kivy photobooth application."""

import subprocess
import sys
import os

def main():
    """Launch the Kivy photobooth application."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    photobooth_script = os.path.join(script_dir, 'photobooth_kivy.py')
    
    try:
        subprocess.run([sys.executable, photobooth_script] + sys.argv[1:], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running photobooth: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPhotobooth stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
