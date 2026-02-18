
import os
import sys
import subprocess
from huggingface_hub import snapshot_download

try:
    print("Downloading/Locating snapshot...")
    model_path = snapshot_download("google/hear")
    print(f"\nSnapshot Path: {model_path}")
    
    print("\n--- Listing Files ---")
    for root, dirs, files in os.walk(model_path):
        for f in files:
            print(os.path.join(root, f).replace(model_path, ""))
            
    print("\n--- saved_model_cli show ---")
    # Using subprocess to call saved_model_cli
    # It usually comes with tensorflow installation.
    # We need to find where it is or run it as a module if possible.
    # Trying `python -m tensorflow.python.tools.saved_model_cli` is often reliable.
    
    cmd = [sys.executable, "-m", "tensorflow.python.tools.saved_model_cli", "show", "--dir", model_path, "--all"]
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

except Exception as e:
    print(f"Error: {e}")
