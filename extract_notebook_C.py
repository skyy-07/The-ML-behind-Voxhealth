
import json
import os

nb_path = r"d:\Building a deep learning model to hear respiratory and pulmonary issues\notebooks\C_multi_task_classifier.ipynb"
script_path = r"d:\Building a deep learning model to hear respiratory and pulmonary issues\run_classifier_C.py"

if os.path.exists(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            code_cells.append("".join(cell['source']))
    
    full_script = "\n\n".join(code_cells)
    
    # Remove any magic commands or non-python syntax if present
    # The dump showed pure python mostly.
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(full_script)
        
    print(f"Extracted notebook code to {script_path}")
else:
    print("Notebook not found.")
