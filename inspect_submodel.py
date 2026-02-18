import os
import tensorflow as tf
from huggingface_hub import snapshot_download

print("Downloading/Finding model...")
path = snapshot_download("google/hear")
sub_path = os.path.join(path, "event_detector", "event_detector_large")

print(f"\n--- Loading Sub-Model: {sub_path} ---")
try:
    # Try tf.saved_model.load first
    model = tf.saved_model.load(sub_path)
    print("✅ tf.saved_model.load SUCCESS")
    print("Signatures:", list(model.signatures.keys()))
    
    # Try inference with dummy data
    infer = model.signatures['serving_default']
    print("Inference function:", infer)
    print("Inputs:", infer.structured_input_signature)
    print("Outputs:", infer.structured_outputs)
    
except Exception as e:
    print(f"❌ tf.saved_model.load FAILED: {e}")

try:
    import tf_keras
    model_k = tf_keras.models.load_model(sub_path)
    print("\n✅ tf_keras.models.load_model SUCCESS")
except Exception as e:
    print(f"\n❌ tf_keras FAILED: {e}")
