import os
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

# Config
DATASETS_ROOT = Path(r"D:\datasets")
PROCESSED_ROOT = DATASETS_ROOT / 'processed'
DATASETS = ['coughvid', 'parkinsons', 'respiratory_sounds', 'coswara']
SR = 16000
DURATION = 2.0  # seconds
NUM_SAMPLES = int(SR * DURATION)

def generate_sine_wave(freq, duration, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Simple sine wave
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return (audio * 32767).astype(np.int16)

print(f"Generating dummy data in {PROCESSED_ROOT}...")

for dataset in DATASETS:
    target_dir = PROCESSED_ROOT / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 5 dummy files per dataset
    print(f"Processing {dataset}...")
    for i in range(5):
        # Vary frequency slightly to make them different
        freq = 440 + (i * 50) 
        audio = generate_sine_wave(freq, DURATION, SR)
        
        fname = f"dummy_{dataset}_{i+1:03d}.wav"
        fpath = target_dir / fname
        
        wav.write(fpath, SR, audio)
        print(f"  Created {fname}")

print("\n✓ Dummy data generation complete.")
print("You can now run Notebook B (B_feature_extraction.ipynb).")
