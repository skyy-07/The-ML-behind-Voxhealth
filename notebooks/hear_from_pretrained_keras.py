


import os
import sys
import numpy as np
import tensorflow as tf
from huggingface_hub import login, from_pretrained_keras, snapshot_download
import tf_keras

# Force UTF-8 stdout if possible, but simpler to just use ascii
# sys.stdout.reconfigure(encoding='utf-8')

print(f"TensorFlow Version: {tf.__version__}")
print(f"TF-Keras Version: {tf_keras.__version__}")

# Login handling
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print(f"Logging in with HF_TOKEN found in environment...")
    login(token=hf_token, new_session=False)
else:
    print("No HF_TOKEN found. Attempting to use existing login...")
    try:
        login(new_session=False)
    except Exception as e:
        print(f"Warning: Login check failed: {e}")

# Method 1: from_pretrained_keras
print("\n--- Method 1: from_pretrained_keras ---")
try:
    model = from_pretrained_keras("google/hear")
    print("[SUCCESS] Method 1 SUCCESS!")
except Exception as e:
    print(f"[FAILED] Method 1 FAILED: {e}")

# Method 2: Manual snapshot + tf_keras.models.load_model
print("\n--- Method 2: snapshot_download + tf_keras.models.load_model ---")
try:
    model_path = snapshot_download("google/hear")
    print(f"Model snapshot path: {model_path}")
    
    # Try loading with tf_keras
    print("Attempting tf_keras.models.load_model...")
    model = tf_keras.models.load_model(model_path)
    print("[SUCCESS] Method 2 SUCCESS!")
    
    # Test inference
    input_shape = (1, 16000)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    print(f"Running inference with shape {input_shape}...")
    embeddings = model.predict(dummy_input)
    print(f"Embeddings shape: {embeddings.shape}")
    
except Exception as e:
    print(f"[FAILED] Method 2 FAILED: {e}")
    
    # Method 3: tf.saved_model.load
    print("\n--- Method 3: tf.saved_model.load ---")
    try:
        loaded = tf.saved_model.load(model_path)
        print("[SUCCESS] Method 3 SUCCESS (Generic SavedModel)!")
        print(f"Signatures: {loaded.signatures.keys()}")
        
        infer = loaded.signatures["serving_default"]
        print("Inference signature obtained.")
        test_input = np.random.randn(1, 16000).astype(np.float32)
        result = infer(tf.constant(test_input))
        print("[SUCCESS] Inference successful (Method 3)!")
        
    except Exception as e3:
        print(f"[FAILED] Method 3 FAILED: {e3}")
