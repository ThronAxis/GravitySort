"""
Patches Section 1 to use wget zip download with CORRECT new repo name:
  ThronAxis/GravitySort  (repo was renamed on GitHub)
"""
import json

NB_PATH = "GravitySort/GravitySort_Kaggle.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

NEW_SEC1 = [
    "import os\n",
    "\n",
    "# Repo: https://github.com/ThronAxis/GravitySort\n",
    "GITHUB_USER = 'ThronAxis'\n",
    "GITHUB_REPO = 'GravitySort'\n",
    "ZIP_URL     = f'https://github.com/{GITHUB_USER}/{GITHUB_REPO}/archive/refs/heads/main.zip'\n",
    "ZIP_PATH    = f'/kaggle/working/{GITHUB_REPO}.zip'\n",
    "EXTRACT_DIR = f'/kaggle/working/{GITHUB_REPO}-main'\n",
    "PROJECT_DIR = EXTRACT_DIR   # GravitySort IS the repo root now\n",
    "\n",
    "if os.path.isdir(PROJECT_DIR):\n",
    "    print(f'Already exists: {PROJECT_DIR}')\n",
    "else:\n",
    "    print(f'Downloading from GitHub...')\n",
    "    r = run(f'wget -q --show-progress -O {ZIP_PATH} \"{ZIP_URL}\"')\n",
    "    if r.returncode != 0:\n",
    "        r = run(f'curl -L -o {ZIP_PATH} \"{ZIP_URL}\"')\n",
    "    assert r.returncode == 0, 'Download FAILED'\n",
    "    print('Extracting...')\n",
    "    run(f'unzip -q {ZIP_PATH} -d /kaggle/working/')\n",
    "\n",
    "assert os.path.isdir(PROJECT_DIR), f'Project not found at {PROJECT_DIR}'\n",
    "print(f'Project root: {PROJECT_DIR}')\n",
    "run(f'find {PROJECT_DIR} -type f | sort')\n"
]

for cell in nb["cells"]:
    if cell.get("id") == "setup-project":
        cell["source"] = NEW_SEC1
        print("Patched setup-project with new repo name: ThronAxis/GravitySort")
        break

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Done.")
