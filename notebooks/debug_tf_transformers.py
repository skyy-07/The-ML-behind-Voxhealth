
import os
import sys
import tensorflow as tf

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

try:
    import transformers
    from transformers import is_tf_available, is_torch_available
    print(f"Transformers version: {transformers.__version__}")
    print(f"is_tf_available: {is_tf_available()}")
    print(f"is_torch_available: {is_torch_available()}")
    
    try:
        from transformers import TFAutoModel
        print("Successfully imported TFAutoModel")
    except ImportError as e:
        print(f"Failed to import TFAutoModel: {e}")
        # Let's inspect what is available
        print("Available in transformers:", dir(transformers))
        
except ImportError as e:
    print(f"Failed to import transformers: {e}")
