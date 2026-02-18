
import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download
import os

try:
    print("Downloading model...")
    model_path = snapshot_download(repo_id="google/hear", repo_type="model")
    print(f"Model path: {model_path}")
    print(f"Contents of model path: {os.listdir(model_path)}")

    print("\nAttempting tf.saved_model.load()...")
    try:
        loaded_model = tf.saved_model.load(model_path)
        print("Success with tf.saved_model.load()!")
        print(f"Signatures: {list(loaded_model.signatures.keys())}")
        
        infer = loaded_model.signatures["serving_default"]
        print("Inference signature obtained.")
        
        test_input = np.random.randn(1, 32000).astype(np.float32)
        print("Running inference...")
        result = infer(tf.constant(test_input))
        print("Inference successful!")
        print(result.keys())
        
    except Exception as e:
        print(f"Failed with tf.saved_model.load(): {e}")

    print("\nAttempting tf.saved_model.load(tags=['serve'])...")
    try:
        loaded_model = tf.saved_model.load(model_path, tags=['serve'])
        print("Success with tf.saved_model.load(tags=['serve'])!")
    except Exception as e:
        print(f"Failed with tags=['serve']: {e}")

except Exception as e:
    print(f"Global failure: {e}")
