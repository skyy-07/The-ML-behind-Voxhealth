
import json
from pathlib import Path

notebook_path = Path(r"d:/Building a deep learning model to hear respiratory and pulmonary issues/notebooks/B_feature_extraction.ipynb")

content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook B: HeAR Feature Extraction\n",
    "## Generate embeddings using HeAR model\n",
    "\n",
    "Loads the HeAR model using direct TensorFlow SavedModel loading to ensure compatibility."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install dependencies\n",
    "%pip install huggingface_hub librosa tensorflow\n",
    "\n",
    "import os\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import librosa\n",
    "import tensorflow as tf\n",
    "from tqdm.notebook import tqdm\n",
    "import json\n",
    "from huggingface_hub import snapshot_download\n",
    "\n",
    "DATASETS_ROOT = Path(r\"D:\\datasets\")\n",
    "PROCESSED_ROOT = DATASETS_ROOT / 'processed'\n",
    "EMBEDDINGS_DIR = DATASETS_ROOT / 'embeddings'\n",
    "EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "TARGET_SR = 16000\n",
    "N_SAMPLES = 32000  # 2 seconds at 16kHz\n",
    "EMBEDDING_DIM = 768 \n",
    "\n",
    "print(f\"Embeddings output: {EMBEDDINGS_DIR}\")\n",
    "print(f\"TensorFlow version: {tf.__version__}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load HeAR Model directly as a SavedModel with tags\n",
    "print(\"Downloading HeAR model from Hugging Face Hub...\")\n",
    "\n",
    "try:\n",
    "    model_path = snapshot_download(repo_id=\"google/hear\", repo_type=\"model\")\n",
    "    print(f\"Model path: {model_path}\")\n",
    "    \n",
    "    print(\"Loading SavedModel directly with tags=['serve']...\")\n",
    "    # Provide the 'serve' tag which is standard for serving signatures\n",
    "    hear_model = tf.saved_model.load(model_path, tags=['serve'])\n",
    "    \n",
    "    # Get the inference function\n",
    "    if 'serving_default' in hear_model.signatures:\n",
    "        inference_fn = hear_model.signatures['serving_default']\n",
    "    else:\n",
    "        # Fallback to first available signature if serving_default is missing\n",
    "        first_key = list(hear_model.signatures.keys())[0]\n",
    "        inference_fn = hear_model.signatures[first_key]\n",
    "        \n",
    "    print(\"\u2713 Model loaded successfully using tf.saved_model.load(tags=['serve'])\")\n",
    "    print(f\"Signatures: {list(hear_model.signatures.keys())}\")\n",
    "except Exception as e:\n",
    "    print(f\"\u26a0 Failed to load model: {e}\")\n",
    "    raise e"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_audio_for_hear(file_path):\n",
    "    \"\"\"Load audio and ensure it's exactly 32000 samples (2s @ 16kHz)\"\"\"\n",
    "    # HeAR expects normalized audio in range [-1, 1]\n",
    "    audio, _ = librosa.load(str(file_path), sr=TARGET_SR, mono=True)\n",
    "    \n",
    "    # Pad or trim to exactly N_SAMPLES\n",
    "    if len(audio) < N_SAMPLES:\n",
    "        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), 'constant')\n",
    "    else:\n",
    "        audio = audio[:N_SAMPLES]\n",
    "        \n",
    "    return audio.astype(np.float32)\n",
    "\n",
    "def extract_embeddings_batch(audio_batch):\n",
    "    \"\"\"Batch extraction using the serving_default signature\"\"\"\n",
    "    # Convert input list to tensor\n",
    "    audio_tensor = tf.convert_to_tensor(audio_batch)\n",
    "    \n",
    "    # Inference: The model expects input key 'x'\n",
    "    # Note: tf.saved_model signatures often return a dict\n",
    "    output_dict = inference_fn(x=audio_tensor)\n",
    "    \n",
    "    # The output key is usually 'output_0'\n",
    "    if 'output_0' in output_dict:\n",
    "        embedding = output_dict['output_0'].numpy()\n",
    "    else:\n",
    "        key = list(output_dict.keys())[0]\n",
    "        embedding = output_dict[key].numpy()\n",
    "        \n",
    "    return embedding\n",
    "\n",
    "print(\"\u2713 Embedding functions ready\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_dataset_embeddings(dataset_name, batch_size=32):\n",
    "    input_dir = PROCESSED_ROOT / dataset_name\n",
    "    if not input_dir.exists():\n",
    "        print(f\"\u26a0 {dataset_name}: Not found\")\n",
    "        return None\n",
    "\n",
    "    wav_files = sorted(input_dir.glob(\"*.wav\"))\n",
    "    if not wav_files:\n",
    "        return None\n",
    "\n",
    "    print(f\"\\nProcessing {dataset_name}: {len(wav_files)} files\")\n",
    "\n",
    "    embeddings_list = []\n",
    "    file_names = []\n",
    "\n",
    "    # Process in batches\n",
    "    for i in tqdm(range(0, len(wav_files), batch_size), desc=f\"Extracting {dataset_name}\"):\n",
    "        batch_files = wav_files[i:i+batch_size]\n",
    "        \n",
    "        # Load batch audio\n",
    "        batch_audio = []\n",
    "        valid_batch_indices = []\n",
    "        \n",
    "        for idx, f in enumerate(batch_files):\n",
    "            try:\n",
    "                audio = load_audio_for_hear(f)\n",
    "                batch_audio.append(audio)\n",
    "                valid_batch_indices.append(idx)\n",
    "            except Exception as e:\n",
    "                print(f\"Error loading {f}: {e}\")\n",
    "        \n",
    "        if not batch_audio:\n",
    "            continue\n",
    "            \n",
    "        try:\n",
    "            batch_embeddings = extract_embeddings_batch(batch_audio)\n",
    "            embeddings_list.append(batch_embeddings)\n",
    "            file_names.extend([batch_files[idx].stem for idx in valid_batch_indices])\n",
    "        except Exception as e:\n",
    "            print(f\"Error batch {i}: {e}\")\n",
    "            continue\n",
    "\n",
    "    if not embeddings_list:\n",
    "        return 0\n",
    "\n",
    "    embeddings = np.vstack(embeddings_list)\n",
    "    output_path = EMBEDDINGS_DIR / f\"{dataset_name}_embeddings.npz\"\n",
    "    np.savez_compressed(output_path, embeddings=embeddings, file_names=file_names)\n",
    "    print(f\"\u2713 {dataset_name}: {embeddings.shape[0]} embeddings saved\")\n",
    "    return embeddings.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Process all datasets\n",
    "datasets = ['coughvid', 'parkinsons', 'respiratory_sounds', 'coswara']\n",
    "results = {}\n",
    "\n",
    "for name in datasets:\n",
    "    results[name] = process_dataset_embeddings(name)\n",
    "\n",
    "summary = {'embedding_dim': EMBEDDING_DIM, 'sample_rate': TARGET_SR, 'datasets': results}\n",
    "with open(EMBEDDINGS_DIR / 'embeddings_summary.json', 'w') as f:\n",
    "    json.dump(summary, f, indent=2)\n",
    "\n",
    "print(f\"\\nTotal embeddings: {sum(v for v in results.values() if v)}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(content, f, indent=1)

print("Notebook updated with tags=['serve'] fix.")
