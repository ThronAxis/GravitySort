"""
Patches Section 1 to use wget zip download instead of git clone.
wget on HTTPS works on Kaggle even when git clone DNS fails.
"""
import json, sys

NB_PATH = "GravitySort/GravitySort_Kaggle.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

NEW_SEC1 = [
    "import os, subprocess\n",
    "\n",
    "GITHUB_USER = 'ThronAxis'\n",
    "GITHUB_REPO = 'GravityShort_Project_file'\n",
    "ZIP_URL     = f'https://github.com/{GITHUB_USER}/{GITHUB_REPO}/archive/refs/heads/main.zip'\n",
    "ZIP_PATH    = f'/kaggle/working/{GITHUB_REPO}.zip'\n",
    "REPO_DIR    = f'/kaggle/working/{GITHUB_REPO}-main'\n",
    "PROJECT_DIR = f'{REPO_DIR}/GravitySort'\n",
    "\n",
    "if os.path.isdir(PROJECT_DIR):\n",
    "    print(f'Already exists: {PROJECT_DIR}')\n",
    "else:\n",
    "    print(f'Downloading repo zip from GitHub...')\n",
    "    r = run(f'wget -q --show-progress -O {ZIP_PATH} \"{ZIP_URL}\"')\n",
    "    if r.returncode != 0:\n",
    "        # fallback: try curl\n",
    "        r = run(f'curl -L -o {ZIP_PATH} \"{ZIP_URL}\"')\n",
    "    assert r.returncode == 0, 'Download FAILED - check internet connection'\n",
    "    print('Extracting...')\n",
    "    run(f'unzip -q {ZIP_PATH} -d /kaggle/working/')\n",
    "\n",
    "assert os.path.isdir(PROJECT_DIR), f'GravitySort not found at {PROJECT_DIR}'\n",
    "print(f'\\nProject root: {PROJECT_DIR}')\n",
    "run(f'find {PROJECT_DIR} -type f | sort')\n"
]

for cell in nb["cells"]:
    if cell.get("id") == "setup-project":
        cell["source"] = NEW_SEC1
        print("Patched: setup-project -> wget zip download")
        break

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done. Commit and push to apply.")
