import json
import os

notebook_path = '/Users/abdussattarsagyngali/Desktop/skin-disease/notebooks/deep-learning-skin-lesion-classification.ipynb'
output_path = '/Users/abdussattarsagyngali/Desktop/skin-disease/extracted_code.py'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code': 
            source = ''.join(cell['source'])
            code_cells.append(source)
            
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n# %% [New Cell]\n\n'.join(code_cells))
        
    print(f"Successfully extracted code to {output_path}")
except Exception as e:
    print(f"Error: {e}")
