
import os
import sys

# Redirect stdout/stderr to files to avoid interleaving issues
sys.stdout = open('debug_output.txt', 'w')
sys.stderr = open('debug_error.txt', 'w')

print(f"Python: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"TensorFlow import failed: {e}")

try:
    import keras
    print(f"Keras: {keras.__version__}")
except ImportError:
    print("Keras not found")

try:
    import tf_keras
    print(f"tf_keras: {tf_keras.__version__}")
except ImportError:
    print("tf_keras not found")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
    print(f"Transformers path: {transformers.__file__}")
    
    from transformers import is_tf_available, is_torch_available, is_flax_available
    print(f"is_tf_available: {is_tf_available()}")
    print(f"is_torch_available: {is_torch_available()}")
    print(f"is_flax_available: {is_flax_available()}")
    
except ImportError as e:
    print(f"Transformers import failed: {e}")

# Check what is in transformers
if 'transformers' in sys.modules:
    print("TFAutoModel in transformers?", 'TFAutoModel' in dir(transformers))
