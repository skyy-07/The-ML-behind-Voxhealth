
import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download
import os

print("Downloading model snapshot...")
model_path = snapshot_download("google/hear")
print(f"Model path: {model_path}")

print("Loading with tf.saved_model.load()...")
try:
    model = tf.saved_model.load(model_path)
    print("✅ Model loaded successfully!")
    print(f"Signatures: {list(model.signatures.keys())}")
    
    infer = model.signatures["serving_default"]
    print("Inference signature obtained.")
    
    # Create dummy input
    # HeAR expects audio. The signature likely expects a specific shape.
    # Let's inspect the input signature if possible or just try (1, 16000)
    print("Inputs:", infer.structured_input_signature)
    print("Outputs:", infer.structured_outputs)
    
    # Try 1 second of audio
    test_input = tf.random.normal([1, 16000])
    print("Running inference...")
    result = infer(test_input)
    print("✅ Inference successful!")
    print("Result keys:", result.keys())
    for k, v in result.items():
        print(f"{k}: {v.shape}")

except Exception as e:
    print(f"❌ Failed: {e}")
