
"""Test script to verify HeAR model loading works"""
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

print(f"TensorFlow version: {tf.__version__}")

try:
    print("Loading HeAR model from Hugging Face Hub...")
    hear_model = from_pretrained_keras("google/hear")
    inference_fn = hear_model.signatures["serving_default"]
    print("✓ Model loaded successfully using from_pretrained_keras")
    print(f"Model type: {type(hear_model)}")
    print(f"Inference function: {type(inference_fn)}")
    
    # Test with random input
    import numpy as np
    test_audio = np.random.randn(1, 32000).astype(np.float32)
    result = inference_fn(x=test_audio)
    print(f"✓ Test inference successful")
    print(f"Output shape: {result['output_0'].shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
