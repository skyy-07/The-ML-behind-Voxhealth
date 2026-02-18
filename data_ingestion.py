"""
HeAR Data Ingestion Pipeline
============================
Automated script to download and extract respiratory disease datasets:
- Coughvid v3 (Kaggle)
- Parkinson's Voice Dataset (Kaggle)
- Respiratory Sound Database (Kaggle)
- Coswara (GitHub)

All datasets are stored in D:\datasets\[dataset_name]
"""

import os
import sys
import subprocess
import shutil
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import requests
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Root directory for all datasets
DATASETS_ROOT = Path(r"D:\datasets")

# Dataset configurations
DATASETS = {
    "coughvid": {
        "kaggle_id": "orvile/coughvid-v3",
        "target_dir": DATASETS_ROOT / "coughvid",
        "source": "kaggle"
    },
    "parkinsons": {
        "kaggle_id": "vikasukani/parkinsons-disease-data-set",
        "target_dir": DATASETS_ROOT / "parkinsons",
        "source": "kaggle"
    },
    "respiratory_sounds": {
        "kaggle_id": "vbookshelf/respiratory-sound-database",
        "target_dir": DATASETS_ROOT / "respiratory_sounds",
        "source": "kaggle"
    },
    "coswara": {
        "github_url": "https://github.com/iiscleap/Coswara-Data.git",
        "target_dir": DATASETS_ROOT / "coswara",
        "source": "github"
    }
}

# Embeddings output directory
EMBEDDINGS_DIR = DATASETS_ROOT / "embeddings"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    """Create all necessary directories."""
    print("\n" + "="*60)
    print("SETTING UP DIRECTORIES")
    print("="*60)
    
    # Create root datasets directory
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created root directory: {DATASETS_ROOT}")
    
    # Create embeddings directory
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created embeddings directory: {EMBEDDINGS_DIR}")
    
    # Create dataset directories
    for name, config in DATASETS.items():
        config["target_dir"].mkdir(parents=True, exist_ok=True)
        print(f"✓ Created dataset directory: {config['target_dir']}")
    
    print("\nAll directories created successfully!")


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    # Check for kaggle.json in current directory first
    local_kaggle = Path("kaggle.json")
    if local_kaggle.exists():
        print(f"✓ Found local kaggle.json: {local_kaggle.absolute()}")
        return True
    
    # Check default location
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print("✓ Kaggle credentials found in default location")
        return True
        
    # Check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("✓ Kaggle credentials found (environment variables)")
        return True
    
    print("\n" + "!"*60)
    print("KAGGLE CREDENTIALS NOT FOUND")
    print("!"*60)
    print("""
To download Kaggle datasets, you need to:
1. Go to https://www.kaggle.com/account
2. Click 'Create New API Token' to download kaggle.json
3. Place kaggle.json in the project root or ~/.kaggle/kaggle.json
""")
    return False


def download_kaggle_dataset(dataset_id: str, target_dir: Path) -> bool:
    """Download and extract a dataset from Kaggle."""
    # Check if data already exists (basic check)
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"\n→ Skipping download for {dataset_id}")
        print(f"  Target directory is not empty: {target_dir}")
        print("  Delete directory to force re-download.")
        return True

    print(f"\n→ Downloading: {dataset_id}")
    
    try:
        # Load local credentials if present
        local_kaggle = Path("kaggle.json")
        if local_kaggle.exists():
            with open(local_kaggle) as f:
                creds = json.load(f)
                os.environ['KAGGLE_USERNAME'] = creds['username']
                os.environ['KAGGLE_KEY'] = creds['key']
        
        # Import kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading to: {target_dir}")
        api.dataset_download_files(
            dataset=dataset_id,
            path=str(target_dir),
            unzip=True,
            quiet=False
        )
        
        print(f"  ✓ Successfully downloaded: {dataset_id}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error downloading {dataset_id}: {str(e)}")
        return False


def clone_github_repo(repo_url: str, target_dir: Path) -> bool:
    """Clone a GitHub repository."""
    # Check if repo already exists
    if target_dir.exists() and (target_dir / ".git").exists():
        print(f"\n→ Verifying {repo_url}")
        print(f"  Repo already exists at: {target_dir}")
        return True

    print(f"\n→ Cloning: {repo_url}")
    
    try:
        # Check if git is available
        result = subprocess.run(["git", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("  ✗ Git is not installed or not in PATH")
            return False
        
        # Remove existing directory if it exists
        if target_dir.exists():
            print(f"  Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)
        
        # Clone the repository
        print(f"  Cloning to: {target_dir}")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  ✗ Git clone failed: {result.stderr}")
            return False
        
        print(f"  ✓ Successfully cloned repository")
        return True
        
    except Exception as e:
        print(f"  ✗ Error cloning repository: {str(e)}")
        return False


def decompress_coswara_data(coswara_dir: Path) -> bool:
    """Decompress Coswara data files."""
    print(f"\n→ Decompressing Coswara data...")
    
    try:
        # Look for compressed data folders (tar.gz files)
        compressed_files = list(coswara_dir.glob("*.tar.gz")) + list(coswara_dir.glob("**/*.tar.gz"))
        
        if not compressed_files:
            # Try running the extract_data.py script if it exists
            extract_script = coswara_dir / "extract_data.py"
            if extract_script.exists():
                print(f"  Running extract_data.py...")
                result = subprocess.run(
                    [sys.executable, str(extract_script)],
                    cwd=str(coswara_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"  ✓ Successfully ran extract_data.py")
                    return True
                else:
                    print(f"  Warning: extract_data.py returned: {result.stderr}")
            
            print(f"  No compressed files found - data may already be extracted")
            return True
        
        # Extract each tar.gz file
        extracted_count = 0
        for compressed_file in tqdm(compressed_files, desc="  Extracting"):
            try:
                with tarfile.open(compressed_file, 'r:gz') as tar:
                    tar.extractall(path=compressed_file.parent)
                extracted_count += 1
            except Exception as e:
                print(f"\n  Warning: Could not extract {compressed_file.name}: {e}")
        
        print(f"  ✓ Extracted {extracted_count} compressed files")
        return True
        
    except Exception as e:
        print(f"  ✗ Error decompressing Coswara data: {str(e)}")
        return False


def get_dataset_stats(dataset_dir: Path) -> dict:
    """Get statistics about a downloaded dataset."""
    stats = {
        "total_files": 0,
        "audio_files": 0,
        "csv_files": 0,
        "total_size_mb": 0
    }
    
    audio_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.webm', '.m4a'}
    
    try:
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                stats["total_files"] += 1
                stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
                
                if file_path.suffix.lower() in audio_extensions:
                    stats["audio_files"] += 1
                elif file_path.suffix.lower() == '.csv':
                    stats["csv_files"] += 1
    except:
        pass
    
    stats["total_size_mb"] = round(stats["total_size_mb"], 2)
    return stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("HeAR DATA INGESTION PIPELINE")
    print("Respiratory Disease Detection Datasets")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Check Kaggle credentials
    kaggle_available = check_kaggle_credentials()
    
    # Track download results
    results = {}
    
    # Download each dataset
    for name, config in DATASETS.items():
        print(f"\n" + "-"*60)
        print(f"PROCESSING: {name.upper()}")
        print("-"*60)
        
        if config["source"] == "kaggle":
            if not kaggle_available:
                print(f"  ⏭ Skipping (Kaggle credentials not available)")
                results[name] = "skipped"
                continue
            
            success = download_kaggle_dataset(
                config["kaggle_id"],
                config["target_dir"]
            )
            results[name] = "success" if success else "failed"
            
        elif config["source"] == "github":
            success = clone_github_repo(
                config["github_url"],
                config["target_dir"]
            )
            
            if success and name == "coswara":
                decompress_coswara_data(config["target_dir"])
            
            results[name] = "success" if success else "failed"
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    for name, config in DATASETS.items():
        status = results.get(name, "unknown")
        status_symbol = "✓" if status == "success" else ("⏭" if status == "skipped" else "✗")
        print(f"\n{status_symbol} {name.upper()}")
        print(f"  Status: {status}")
        print(f"  Location: {config['target_dir']}")
        
        if status == "success":
            stats = get_dataset_stats(config['target_dir'])
            print(f"  Files: {stats['total_files']} ({stats['audio_files']} audio, {stats['csv_files']} CSV)")
            print(f"  Size: {stats['total_size_mb']} MB")
    
    print(f"\n✓ Embeddings will be stored in: {EMBEDDINGS_DIR}")
    
    print("\n" + "="*60)
    print("DATA INGESTION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run Notebook A (preprocessing.ipynb) to standardize audio to 16kHz mono WAV")
    print("2. Run Notebook B (feature_extraction.ipynb) to generate HeAR embeddings")
    print("3. Run Notebook C (multi_task_classifier.ipynb) to train disease classifiers")
    print("4. Run validation.py for cross-validation evaluation")


if __name__ == "__main__":
    main()
