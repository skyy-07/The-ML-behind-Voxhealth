
"""Test HeAR model loading with different methods"""
import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download

print(f"TensorFlow version: {tf.__version__}")

# Method 1: Download model files and load with tf.saved_model.load
try:
    print("\n=== Method 1: Using tf.saved_model.load ===")
    model_path = snapshot_download(repo_id="google/hear", repo_type="model")
    print(f"Model downloaded to: {model_path}")
    
    hear_model = tf.saved_model.load(model_path)
    print(f"✓ Model loaded successfully")
    print(f"Available signatures: {list(hear_model.signatures.keys())}")
    
    inference_fn = hear_model.signatures["serving_default"]
    
    # Test inference
    test_audio = np.random.randn(1, 32000).astype(np.float32)
    result = inference_fn(x=tf.constant(test_audio))
    print(f"✓ Test inference successful")
    print(f"Output shape: {result['output_0'].shape}")
    
except Exception as e:
    print(f"✗ Method 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Method 2: Try keras.models.load_model
try:
    print("\n=== Method 2: Using keras.models.load_model ===")
    import tf_keras
    model_path = snapshot_download(repo_id="google/hear", repo_type="model")
    hear_model = tf_keras.models.load_model(model_path)
    print(f"✓ Model loaded with keras")
    
except Exception as e:
    print(f"✗ Method 2 failed: {e}")
