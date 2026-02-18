
import sys
import os

sys.stdout = open('debug_output_v3.txt', 'w')
sys.stderr = open('debug_error_v3.txt', 'w')

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print(f"Dir(transformers): {dir(transformers)[:20]}...")
    
    # Check if utils available
    try:
        from transformers import utils
        print("Utils available")
        print(f"Dir(utils): {dir(utils)[:20]}...")
    except ImportError:
        print("Utils not available")

    # Try to import TFAutoModel manually
    try:
        from transformers.models.auto import TFAutoModel
        print("TFAutoModel found in models.auto")
    except ImportError:
        print("TFAutoModel not found in models.auto")

except ImportError as e:
    print(f"Failed to import transformers: {e}")
