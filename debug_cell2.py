import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm.notebook import tqdm
import json
import pickle

DATASETS_ROOT = Path(r"D:\datasets")
PROCESSED_ROOT = DATASETS_ROOT / 'processed'
EMBEDDINGS_DIR = DATASETS_ROOT / 'embeddings'
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
N_SAMPLES = 32000  # 2 seconds at 16kHz
EMBEDDING_DIM = 512

print(f"Embeddings output: {EMBEDDINGS_DIR}")
