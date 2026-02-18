import os
import sys
import tensorflow as tf

print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")

# Ensure transformers is installed
try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Installing transformers...")
    os.system(f"{sys.executable} -m pip install transformers")
    import transformers

from transformers import TFAutoModel, AutoFeatureExtractor

print("\nAttempting to load 'google/hear' using Transformers...")
try:
    # HeAR is a ViT-based model, usually loaded with AutoModel
    model = TFAutoModel.from_pretrained("google/hear", trust_remote_code=True)
    print("\n✅ SUCCESS: Model loaded successfully!")
    print(f"Model config: {model.config}")
except Exception as e:
    print(f"\n❌ FAILED to load root model: {e}")
    
    print("\nAttempting to load 'event_detector_large' sub-model...")
    sub_model_path = os.path.join("google", "hear", "event_detector", "event_detector_large")
    # Actually need full path from snapshot
    # Find the snapshot dir again
    from huggingface_hub import snapshot_download
    path = snapshot_download("google/hear")
    sub_path = os.path.join(path, "event_detector", "event_detector_large")
    
    try:
        import tf_keras
        model = tf_keras.models.load_model(sub_path)
        print("\n✅ SUCCESS: Sub-model loaded!")
    except Exception as sub_e:
        print(f"\n❌ FAILED to load sub-model: {sub_e}")

