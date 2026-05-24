import os
import glob

base_dir = r"C:\Users\User\Desktop\VALIDAR HISTORICOS"
py_files = glob.glob(os.path.join(base_dir, "**", "*.py"), recursive=True)

print(f"Searching {len(py_files)} python files for references...")
for fpath in py_files:
    try:
        with open(fpath, 'r', encoding='latin1') as f:
            content = f.read()
            if "historicos" in content.lower():
                print(f"Found 'historicos' in: {fpath}")
            if "historico" in content.lower():
                print(f"Found 'historico' in: {fpath}")
    except Exception as e:
        pass
print("Search done.")
