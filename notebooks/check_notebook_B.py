
import json
import os

nb_path = r"d:\Building a deep learning model to hear respiratory and pulmonary issues\notebooks\B_feature_extraction.ipynb"
if os.path.exists(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Cells in {nb_path}:")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            print(f"\n--- Cell {i} ---")
            print("".join(cell['source']))
else:
    print("Notebook not found.")
