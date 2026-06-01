import json

with open('06_intel_image_classification.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find import cells and cells storing results
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    
    source = ''.join(cell.get('source', []))
    
    # Find first imports
    if i < 5 and 'import torch' in source:
        print(f"Cell {i}: First torch import")
        print(source[:200])
        print()
    
    # Find results storage cells
    if 'results[' in source:
        print(f"Cell {i} (ID: {cell.get('id')}): Stores results")
        print(source)
        print()
