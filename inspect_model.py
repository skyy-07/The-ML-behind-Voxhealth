import os
import subprocess
from huggingface_hub import snapshot_download

print("Downloading/Finding model...")
model_path = snapshot_download("google/hear")
print(f"Model Path: {model_path}")

try:
    print("\n--- Running saved_model_cli ---")
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "tensorflow.python.tools.saved_model_cli", "show", "--dir", model_path, "--all"],
        capture_output=True,
        text=True
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if "MetaGraphDef with tag-set" not in result.stdout:
        print("\n⚠️  No MetaGraphDefs found in output!")
        # Try inspecting the PB file directly
        pb_path = os.path.join(model_path, "saved_model.pb")
        if os.path.exists(pb_path):
             print(f"saved_model.pb exists ({os.path.getsize(pb_path)} bytes)")
             with open(pb_path, "rb") as f:
                 header = f.read(16)
                 print(f"Header: {header}")
except Exception as e:
    print(f"Error: {e}")
