
import json
import os

nb_path = r"d:\Building a deep learning model to hear respiratory and pulmonary issues\notebooks\C_multi_task_classifier.ipynb"
if os.path.exists(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Cells in {nb_path}:")

    with open('notebook_c_dump.txt', 'w', encoding='utf-8') as outfile:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                outfile.write(f"\n--- Cell {i} ---\n")
                outfile.write("".join(cell['source']))
    print("Dumped to notebook_c_dump.txt")
else:
    print("Notebook not found.")
