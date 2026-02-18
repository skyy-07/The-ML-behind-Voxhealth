# HeAR - Health Acoustic Representations Pipeline

## Respiratory Disease Detection using Deep Learning

This project implements a machine learning pipeline for detecting respiratory diseases using Google's Health Acoustic Representations (HeAR) model.

### Diseases Detected
- **Tuberculosis/COVID-19** - Using cough sounds (Coughvid + Coswara)
- **Parkinson's Disease** - Using voice recordings
- **Pulmonary Anomalies** - Using lung sounds (crackles, wheezes)

---

## Project Structure

```
├── requirements.txt              # Python dependencies
├── data_ingestion.py            # Download datasets from Kaggle/GitHub
├── validation.py                # Cross-validation evaluation script
├── notebooks/
│   ├── A_preprocessing.ipynb    # Audio standardization (16kHz mono WAV)
│   ├── B_feature_extraction.ipynb  # HeAR embedding generation
│   └── C_multi_task_classifier.ipynb  # Multi-task disease classifier
└── README.md
```

## Data Directories (D:\datasets)

```
D:\datasets\
├── coughvid/              # Coughvid v3 dataset
├── parkinsons/            # Parkinson's voice dataset  
├── respiratory_sounds/    # Respiratory sound database
├── coswara/              # Coswara COVID-19 dataset
├── processed/            # Preprocessed 16kHz WAV clips
├── embeddings/           # HeAR 512-dim embeddings
├── models/               # Trained classifiers
└── validation_results/   # Evaluation metrics
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token" to download `kaggle.json`
3. Place in `C:\Users\<username>\.kaggle\kaggle.json`

### 3. Configure Hugging Face

```python
# In Python or Jupyter
from huggingface_hub import notebook_login
notebook_login()  # Enter your HF token
```

---

## Usage

### Step 1: Data Ingestion

```bash
python data_ingestion.py
```

Downloads all datasets:
- **Coughvid v3**: `orvile/coughvid-v3`
- **Parkinson's**: `vikasukani/parkinsons-disease-data-set`
- **Respiratory Sounds**: `vbookshelf/respiratory-sound-database`
- **Coswara**: GitHub clone + extraction

### Step 2: Preprocessing (Notebook A)

Open `notebooks/A_preprocessing.ipynb` and run all cells.

Converts audio to HeAR format:
- Sample rate: 16kHz
- Channels: Mono
- Duration: 2-second clips
- Format: WAV (PCM 16-bit)

### Step 3: Feature Extraction (Notebook B)

Open `notebooks/B_feature_extraction.ipynb` and run all cells.

- Loads HeAR model from Hugging Face
- Generates 512-dimensional embeddings
- Saves to `D:\datasets\embeddings\`

### Step 4: Train Classifiers (Notebook C)

Open `notebooks/C_multi_task_classifier.ipynb` and run all cells.

Trains three classifier heads:
1. TB/COVID Classifier (binary)
2. Parkinson's Classifier (binary)
3. Pulmonary Anomaly Classifier (multi-class)

### Step 5: Validation

```bash
python validation.py
```

Runs 5-fold cross-validation and reports:
- F1 Score (weighted)
- AUC-ROC

---

## HeAR Model Specifications

| Property | Value |
|----------|-------|
| Architecture | ViT-L (Vision Transformer) |
| Input | 2-second audio @ 16kHz mono |
| Input Shape | (batch, 32000) |
| Output | 512-dimensional embedding |
| Training Data | 300M+ audio clips (174k hours) |

### Key Features
- Optimized for health acoustics (coughs, breathing)
- Superior device generalization
- Data-efficient for downstream tasks
- Pre-trained with masked auto-encoding

---

## References

- [HeAR Paper (arXiv)](https://arxiv.org/abs/2403.02522)
- [HeAR on Hugging Face](https://huggingface.co/google/hear)
- [Google Health HeAR Documentation](https://developers.google.com/health-ai-developer-foundations/hear)
- [HeAR GitHub Repository](https://github.com/Google-Health/hear)

---

## License

This project uses the HeAR model under the [Health AI Developer Foundations License](https://developers.google.com/health-ai-developer-foundations/terms).

Code in this repository is provided for research purposes.
