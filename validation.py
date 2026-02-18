"""
HeAR Validation Script
======================
Cross-validation evaluation of disease classifiers using HeAR embeddings.
Calculates F1-score and AUC-ROC for each disease category:
- TB/COVID Detection
- Parkinson's Detection  
- Pulmonary Anomaly Detection
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer, f1_score, roc_auc_score, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

DATASETS_ROOT = Path(r"D:\datasets")
EMBEDDINGS_DIR = DATASETS_ROOT / "embeddings"
MODELS_DIR = DATASETS_ROOT / "models"
RESULTS_DIR = DATASETS_ROOT / "validation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42


# ============================================================================
# Utility Functions
# ============================================================================

def load_embeddings(dataset_name):
    """Load embeddings from NPZ file."""
    path = EMBEDDINGS_DIR / f"{dataset_name}_embeddings.npz"
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    return data['embeddings'], data['file_names']


def load_labels_from_metadata(dataset_name, embeddings_dir):
    """Load labels from dataset metadata files."""
    # This would be customized based on actual metadata structure
    # For now, returns synthetic labels for demonstration
    return None


def create_synthetic_labels(n_samples, n_classes=2):
    """Create synthetic labels for demonstration purposes."""
    return np.random.randint(0, n_classes, n_samples)


def get_classifier(model_type='mlp'):
    """Get classifier instance."""
    if model_type == 'mlp':
        return MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    elif model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE
        )
    else:
        return LogisticRegression(
            max_iter=1000,
            C=0.1,
            random_state=RANDOM_STATE
        )


# ============================================================================
# Cross-Validation Functions
# ============================================================================

def cross_validate_classifier(X, y, model_type='mlp', n_folds=N_FOLDS):
    """
    Perform stratified k-fold cross-validation.
    
    Returns:
        dict: Results including F1 scores and AUC-ROC scores per fold
    """
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    results = {
        'f1_scores': [],
        'auc_scores': [],
        'fold_reports': []
    }
    
    print(f"  Running {n_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_encoded)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # Train classifier
        clf = get_classifier(model_type)
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)
        
        # Calculate metrics
        f1 = f1_score(y_val, y_pred, average='weighted')
        results['f1_scores'].append(f1)
        
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_val, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')
            results['auc_scores'].append(auc)
        except Exception as e:
            results['auc_scores'].append(None)
        
        print(f"    Fold {fold+1}: F1={f1:.4f}, AUC={results['auc_scores'][-1]:.4f if results['auc_scores'][-1] else 'N/A'}")
    
    # Calculate summary statistics
    results['f1_mean'] = np.mean(results['f1_scores'])
    results['f1_std'] = np.std(results['f1_scores'])
    
    valid_aucs = [a for a in results['auc_scores'] if a is not None]
    if valid_aucs:
        results['auc_mean'] = np.mean(valid_aucs)
        results['auc_std'] = np.std(valid_aucs)
    else:
        results['auc_mean'] = None
        results['auc_std'] = None
    
    return results


def validate_tb_classifier():
    """Validate TB/COVID classifier using Coughvid + Coswara."""
    print("\n" + "="*60)
    print("VALIDATING TB/COVID CLASSIFIER")
    print("="*60)
    
    X_list, y_list = [], []
    
    for dataset in ['coughvid', 'coswara']:
        emb, files = load_embeddings(dataset)
        if emb is not None:
            print(f"  Loaded {dataset}: {emb.shape}")
            X_list.append(emb)
            # Note: Replace with actual label loading
            y_list.append(create_synthetic_labels(len(emb), n_classes=2))
    
    if not X_list:
        print("  No embeddings available")
        return None
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    print(f"  Total samples: {len(X)}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    results = cross_validate_classifier(X, y, model_type='mlp')
    
    print(f"\n  Results:")
    print(f"    F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    if results['auc_mean']:
        print(f"    AUC-ROC:  {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
    
    return results


def validate_parkinsons_classifier():
    """Validate Parkinson's classifier."""
    print("\n" + "="*60)
    print("VALIDATING PARKINSON'S CLASSIFIER")
    print("="*60)
    
    emb, files = load_embeddings('parkinsons')
    if emb is None:
        print("  No embeddings available")
        return None
    
    print(f"  Loaded parkinsons: {emb.shape}")
    
    X = emb
    y = create_synthetic_labels(len(X), n_classes=2)
    
    print(f"  Total samples: {len(X)}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    results = cross_validate_classifier(X, y, model_type='mlp')
    
    print(f"\n  Results:")
    print(f"    F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    if results['auc_mean']:
        print(f"    AUC-ROC:  {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
    
    return results


def validate_pulmonary_classifier():
    """Validate Pulmonary anomaly classifier."""
    print("\n" + "="*60)
    print("VALIDATING PULMONARY ANOMALY CLASSIFIER")
    print("="*60)
    
    emb, files = load_embeddings('respiratory_sounds')
    if emb is None:
        print("  No embeddings available")
        return None
    
    print(f"  Loaded respiratory_sounds: {emb.shape}")
    
    X = emb
    # Multi-class: 0=normal, 1=crackles, 2=wheezes, 3=both
    y = create_synthetic_labels(len(X), n_classes=4)
    
    print(f"  Total samples: {len(X)}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    results = cross_validate_classifier(X, y, model_type='mlp')
    
    print(f"\n  Results:")
    print(f"    F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    if results['auc_mean']:
        print(f"    AUC-ROC:  {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_results(all_results):
    """Plot validation results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 Scores
    tasks = []
    f1_means = []
    f1_stds = []
    
    for task, results in all_results.items():
        if results:
            tasks.append(task.replace('_', '\n'))
            f1_means.append(results['f1_mean'])
            f1_stds.append(results['f1_std'])
    
    if tasks:
        x = np.arange(len(tasks))
        axes[0].bar(x, f1_means, yerr=f1_stds, capsize=5, color='steelblue', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tasks)
        axes[0].set_ylabel('F1 Score')
        axes[0].set_title('F1 Score by Disease Category')
        axes[0].set_ylim(0, 1)
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
        axes[0].legend()
    
    # AUC Scores
    auc_means = []
    auc_stds = []
    auc_tasks = []
    
    for task, results in all_results.items():
        if results and results.get('auc_mean'):
            auc_tasks.append(task.replace('_', '\n'))
            auc_means.append(results['auc_mean'])
            auc_stds.append(results['auc_std'])
    
    if auc_tasks:
        x = np.arange(len(auc_tasks))
        axes[1].bar(x, auc_means, yerr=auc_stds, capsize=5, color='coral', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(auc_tasks)
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_title('AUC-ROC by Disease Category')
        axes[1].set_ylim(0, 1)
        axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'validation_results.png', dpi=150)
    plt.show()
    print(f"\nPlot saved to: {RESULTS_DIR / 'validation_results.png'}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run complete validation pipeline."""
    print("\n" + "="*60)
    print("HeAR CROSS-VALIDATION EVALUATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Folds: {N_FOLDS}")
    print("="*60)
    
    all_results = {}
    
    # Validate each classifier
    all_results['tb_covid'] = validate_tb_classifier()
    all_results['parkinsons'] = validate_parkinsons_classifier()
    all_results['pulmonary'] = validate_pulmonary_classifier()
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    summary = {}
    for task, results in all_results.items():
        if results:
            summary[task] = {
                'f1_mean': results['f1_mean'],
                'f1_std': results['f1_std'],
                'auc_mean': results.get('auc_mean'),
                'auc_std': results.get('auc_std')
            }
            print(f"\n{task.upper()}:")
            print(f"  F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
            if results.get('auc_mean'):
                print(f"  AUC-ROC:  {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
        else:
            print(f"\n{task.upper()}: No results available")
    
    # Save results
    results_path = RESULTS_DIR / 'validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Plot results
    try:
        plot_results(all_results)
    except Exception as e:
        print(f"Note: Could not generate plot: {e}")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    main()
