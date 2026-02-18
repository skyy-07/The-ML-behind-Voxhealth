
import json
import os

nb_path = r"d:\Building a deep learning model to hear respiratory and pulmonary issues\notebooks\C_multi_task_classifier.ipynb"
OUTPUT_PATH = nb_path  # Overwrite or separate file

if os.path.exists(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Create new cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize models to prevent NameError if training cells are skipped or fail\n",
            "tb_clf = None\n",
            "pd_clf = None\n",
            "pulm_clf = None\n",
            "tb_results = None\n",
            "pd_results = None\n",
            "pulm_results = None\n"
        ]
    }
    
    # Check if cell already exists to avoid duplicates
    existing_source = "".join(nb['cells'][2]['source']) if len(nb['cells']) > 2 else ""
    if "tb_clf = None" in existing_source:
        print("Initialization cell already exists.")
    else:
        # Insert after imports (Cell 1 usually)
        # Based on dump, Cell 0 is empty? Cell 1 is imports.
        # Let's insert at index 2.
        nb['cells'].insert(2, new_cell)
        
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        
        print(f"Successfully added initialization cell to {OUTPUT_PATH}")

else:
    print("Notebook not found.")
