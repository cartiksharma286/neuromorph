import os
import json
import sys

def load_structure(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def init_dhf(config):
    root = config.get('root', 'DHF')
    if not os.path.exists(root):
        os.makedirs(root)
        print(f"Created root directory: {root}")
    
    structure = config.get('structure', {})
    for folder, files in structure.items():
        folder_path = os.path.join(root, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")
        
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(f"# {file_name.replace('_', ' ').replace('.md', '')}\n\nTODO: Add content here.\n")
                print(f"Created file: {file_path}")

def validate_dhf(config):
    root = config.get('root', 'DHF')
    structure = config.get('structure', {})
    all_good = True
    
    if not os.path.exists(root):
        print(f"MISSING: Root directory {root}")
        return False

    for folder, files in structure.items():
        folder_path = os.path.join(root, folder)
        if not os.path.exists(folder_path):
            print(f"MISSING: Directory {folder_path}")
            all_good = False
            continue
            
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                print(f"MISSING: File {file_path}")
                all_good = False
            else:
                # Check for TODO
                with open(file_path, 'r') as f:
                    content = f.read()
                    if "TODO" in content:
                        print(f"WARNING: {file_path} contains TODOs")
    
    if all_good:
        print("DHF Validation Passed (Structure exists).")
    else:
        print("DHF Validation Failed.")

if __name__ == "__main__":
    config_path = "iec_13485_compliance/structure.json"
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        sys.exit(1)
        
    config = load_structure(config_path)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        init_dhf(config)
    else:
        validate_dhf(config)
