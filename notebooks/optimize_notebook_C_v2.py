
import json
import os

NOTEBOOK_PATH = r"d:\Building a deep learning model to hear respiratory and pulmonary issues\notebooks\C_multi_task_classifier.ipynb"
MODIFIED_NOTEBOOK_PATH = NOTEBOOK_PATH

def optimize_notebook(nb_content):
    # Iterate through cells and modify specific logic
    for cell in nb_content['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        source = "".join(cell['source'])
        
        # 1. Modify TB Classifier (Coughvid/Coswara)
        if "def train_tb_classifier():" in source:
            print("Optimizing TB Classifier...")
            # Reduce dataset size for speed if it's too large, but here we want performance.
            # The issue might be KMeans hanging on large data or just taking time.
            # Let's add a MiniBatchKMeans for speed on large datasets
            
            new_source = source.replace(
                "from sklearn.cluster import KMeans",
                "from sklearn.cluster import MiniBatchKMeans"
            )
            new_source = new_source.replace(
                "kmeans = KMeans(n_clusters=2, random_state=42)",
                "kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=1024)"
            )
            cell['source'] = new_source.splitlines(keepends=True)
            
    return nb_content

if os.path.exists(NOTEBOOK_PATH):
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    optimized_nb = optimize_notebook(nb)
    
    with open(MODIFIED_NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(optimized_nb, f, indent=1)
        
    print(f"✅ Notebook optimized (MiniBatchKMeans) and saved to {MODIFIED_NOTEBOOK_PATH}")
else:
    print(f"❌ Notebook {NOTEBOOK_PATH} not found")
