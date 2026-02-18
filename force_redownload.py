import shutil
import os
from huggingface_hub import snapshot_download
import tensorflow as tf

print("Clearing cache for google/hear...")
try:
    # Find cache dir
    path = snapshot_download("google/hear")
    print(f"Current Path: {path}")
    
    # Go up to the repo dir
    repo_dir = os.path.dirname(os.path.dirname(path))
    # Or just delete the snapshot
    if os.path.exists(path):
        print(f"Deleting snapshot: {path}")
        shutil.rmtree(path)
        print("Deleted.")
    
    # Also check 'blobs' if relevant, but snapshot removal forces re-link/download attempts usually
    
except Exception as e:
    print(f"Error clearing cache: {e}")

print("\nRe-downloading model (Force)...")
new_path = snapshot_download("google/hear", force_download=True)
print(f"New Path: {new_path}")

print("\nVerifying 'saved_model.pb' size...")
pb_path = os.path.join(new_path, "saved_model.pb")
if os.path.exists(pb_path):
    print(f"Size: {os.path.getsize(pb_path)/(1024*1024):.2f} MB")
else:
    print("❌ saved_model.pb missing!")

print("\nAttempting load again...")
try:
    model = tf.saved_model.load(new_path)
    print("✅ Load SUCCESS")
except Exception as e:
    print(f"❌ Load FAILED: {e}")
