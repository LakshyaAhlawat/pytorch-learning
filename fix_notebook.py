#!/usr/bin/env python3
"""
Fix the kernel crash in 06_intel_image_classification.ipynb by adding memory management
"""

import json
import sys

def fix_notebook():
    notebook_path = '06_intel_image_classification.ipynb'
    
    print(f"Opening {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Total cells: {len(nb['cells'])}")
    
    # Step 1: Find first code cell with imports and add 'import gc' if missing
    print("\n[Step 1] Adding 'import gc' to first imports cell...")
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            source_lines = cell.get('source', [])
            source_str = ''.join(source_lines)
            
            if 'import torch' in source_str and 'import gc' not in source_str:
                print(f"  Found first torch import at cell {i}")
                
                # Add gc import after other imports
                if isinstance(source_lines, list):
                    # Find last import line
                    last_import_idx = 0
                    for j, line in enumerate(source_lines):
                        if line.strip().startswith('import') or line.strip().startswith('from'):
                            last_import_idx = j
                    
                    # Insert gc import after last import
                    source_lines.insert(last_import_idx + 1, 'import gc\n')
                    cell['source'] = source_lines
                    print(f"  Added 'import gc' after line {last_import_idx}")
                
                break
    
    # Step 2: Find the crash cell (ID: 811131ca) and add memory cleanup BEFORE storing results
    print("\n[Step 2] Modifying crash cell (ID: 811131ca) to add memory cleanup...")
    for i, cell in enumerate(nb['cells']):
        if cell.get('id') == '811131ca':
            print(f"  Found crash cell at index {i}")
            
            source_lines = cell.get('source', [])
            source_str = ''.join(source_lines)
            
            # The cell currently is:
            # if "results" not in globals() or not isinstance(results, dict):
            #   results = {}
            # results["model_0"] = model_0_results
            
            # We need to add memory cleanup BEFORE the results assignment
            new_source = [
                'if "results" not in globals() or not isinstance(results, dict):\n',
                '  results = {}\n',
                '\n',
                '# Clear model from memory to free GPU/RAM\n',
                'model_0.cpu() if torch.cuda.is_available() else None\n',
                'del model_0\n',
                'gc.collect()\n',
                'if torch.cuda.is_available():\n',
                '  torch.cuda.empty_cache()\n',
                '\n',
                'results["model_0"] = model_0_results\n'
            ]
            
            cell['source'] = new_source
            print("  Added memory cleanup code")
            break
    
    # Step 3: Find similar cells for other models (model_1, model_2, etc.) and add same cleanup
    print("\n[Step 3] Finding and fixing similar cells for other models...")
    model_results_indices = []
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            source_str = ''.join(cell.get('source', []))
            # Find cells that store model results
            if 'results["model_' in source_str and '= model_' in source_str:
                model_results_indices.append(i)
                print(f"  Found model storage cell at index {i}")
    
    # Fix each model storage cell
    for cell_idx in model_results_indices:
        cell = nb['cells'][cell_idx]
        source_lines = cell.get('source', [])
        source_str = ''.join(source_lines)
        
        # Extract model index (model_0, model_1, etc.)
        import re
        match = re.search(r'results\["model_(\d+)"\]', source_str)
        if match:
            model_num = match.group(1)
            
            # Skip if already fixed (811131ca)
            if cell.get('id') == '811131ca':
                continue
            
            print(f"  Fixing cell for model_{model_num}...")
            
            # Add cleanup before each results assignment
            new_lines = []
            for line in source_lines:
                if f'results["model_{model_num}"]' in line and '=' in line and 'model_{model_num}_results' in line:
                    # Add cleanup code before this line
                    new_lines.extend([
                        f'# Clear model_{model_num} from memory to free GPU/RAM\n',
                        f'model_{model_num}.cpu() if torch.cuda.is_available() else None\n',
                        f'del model_{model_num}\n',
                        'gc.collect()\n',
                        'if torch.cuda.is_available():\n',
                        '  torch.cuda.empty_cache()\n',
                        '\n'
                    ])
                new_lines.append(line)
            
            cell['source'] = new_lines
    
    # Save the modified notebook
    print(f"\n[Saving] Writing modified notebook to {notebook_path}...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("\n✓ Notebook fixed successfully!")
    print("\nChanges made:")
    print("1. Added 'import gc' to first imports cell")
    print("2. Added memory cleanup code to all model storage cells:")
    print("   - Move model to CPU")
    print("   - Delete model object")
    print("   - Run garbage collection")
    print("   - Clear CUDA cache if available")

if __name__ == '__main__':
    try:
        fix_notebook()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
