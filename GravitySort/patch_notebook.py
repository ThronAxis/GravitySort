"""Fix PROJECT_DIR path in all notebook cells - add /GravitySort subfolder"""
import json

NB_PATH = "GravitySort/GravitySort_Kaggle.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

OLD = "/kaggle/working/GravitySort-main'"
NEW = "/kaggle/working/GravitySort-main/GravitySort'"

fixed = 0
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_lines = []
        changed = False
        for line in cell["source"]:
            if OLD in line and "GravitySort-main/GravitySort" not in line:
                line = line.replace(OLD, NEW)
                changed = True
            new_lines.append(line)
        if changed:
            cell["source"] = new_lines
            fixed += 1
            print("Fixed cell:", cell.get("id"))

print("Total cells fixed:", fixed)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done.")
