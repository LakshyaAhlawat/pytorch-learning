"""
Script to fix kernel crash in 06_intel_image_classification.ipynb
by adding memory management and garbage collection
"""
import json
import re

def fix_notebook():
    # Load the notebook
    with open('06_intel_image_classification.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Add garbage collection import at the beginning if not present
    first_code_cell_found = False
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code' and not first_code_cell_found:
            source = ''.join(cell.get('source', []))
            if 'import' in source and 'import torch' in source:
                if 'import gc' not in source:
                    # Add gc import after other imports
                    lines = source.split('\n')
                    import_idx = 0
                    for i, line in enumerate(lines):
                        if 'from torch' in line or 'import torch' in line:
                            import_idx = i + 1
                    
                    # Insert gc import
                    lines.insert(import_idx, 'import gc')
                    cell['source'] = [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]
                    first_code_cell_found = True
                    break
    
    # Find and modify cells that train models and store results
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
            
        source = ''.join(cell.get('source', []))
        
        # Find cells storing model results globally (e.g., results["model_0"] = ...)
        if re.search(r'results\["model_\d+"\]\s*=', source):
            print(f"Found results storage cell at index {i}")
            
            # Add memory cleanup before storing results
            # Add model to CPU and delete if on GPU
            lines = source.split('\n')
            
            # Find the line with results["model_X"] = 
            for j, line in enumerate(lines):
                if re.search(r'results\["model_\d+"\]\s*=', line):
                    # Extract model variable name
                    match = re.search(r'results\["model_(\d+)"\]\s*=\s*(\w+)', line)
                    if match:
                        model_idx = match.group(1)
                        var_name = match.group(2)
                        
                        # Insert cleanup code before storing results
                        cleanup_code = [
                            f"# Clean up model_0_{model_idx} to free memory",
                            f"model_{model_idx}.cpu() if torch.cuda.is_available() else None",
                            f"del model_{model_idx}",
                            "gc.collect()",
                            "if torch.cuda.is_available():",
                            "    torch.cuda.empty_cache()",
                            ""
                        ]
                        lines = lines[:j] + cleanup_code + lines[j:]
                    break
            
            cell['source'] = [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]
    
    # Save the modified notebook
    with open('06_intel_image_classification.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook fixed successfully!")

if __name__ == '__main__':
    fix_notebook()
