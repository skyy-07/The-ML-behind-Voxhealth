import sys
import os
import numpy as np

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✓ {module_name} is installed")
        return True
    except ImportError as e:
        print(f"✗ {module_name} is MISSING: {e}")
        return False

print("="*40)
print("Environment Verification")
print(f"Python: {sys.version.split()[0]}")
print("="*40)

required_modules = [
    'tensorflow',
    'librosa',
    'soundfile',
    'pandas',
    'numpy',
    'sklearn',
    'huggingface_hub',
    'kaggle'
]

all_ok = True
for mod in required_modules:
    if not check_import(mod):
        all_ok = False

print("-" * 40)
if all_ok:
    print("✓ Environment looks good! You can proceed with data ingestion.")
else:
    print("⚠ Some dependencies are missing. Run: pip install -r requirements.txt")
print("-" * 40)
