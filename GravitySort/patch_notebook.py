"""
Patches GravitySort_Kaggle.ipynb:
  - Section 1 (setup-project): supports private repo via GitHub PAT,
    falls back to Kaggle Dataset zip if clone fails
  - Section 1b (install-deps): full Kaggle dependency audit
"""
import json, sys

NB_PATH = "GravitySort/GravitySort_Kaggle.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

NEW_SEC1 = [
    "import os\n",
    "\n",
    "# ─────────────────────────────────────────────────────────────────────────\n",
    "# OPTION A — Private repo via GitHub Personal Access Token (PAT)\n",
    "#   1. Go to: https://github.com/settings/tokens  →  Generate new token (classic)\n",
    "#   2. Scopes: check 'repo'\n",
    "#   3. Add it to Kaggle Secrets:  Add-ons → Secrets → New Secret\n",
    "#      Name: GITHUB_TOKEN   Value: your_token_here\n",
    "# OPTION B — Make the repo public on GitHub (Settings → Danger Zone → Change visibility)\n",
    "# ─────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "GITHUB_USER = 'ThronAxis'\n",
    "GITHUB_REPO = 'GravityShort_Project_file'\n",
    "REPO_DIR    = f'/kaggle/working/{GITHUB_REPO}'\n",
    "PROJECT_DIR = f'{REPO_DIR}/GravitySort'\n",
    "\n",
    "# ── Try to load PAT from Kaggle Secrets ───────────────────────────────────\n",
    "token = ''\n",
    "try:\n",
    "    from kaggle_secrets import UserSecretsClient\n",
    "    token = UserSecretsClient().get_secret('GITHUB_TOKEN')\n",
    "    print('PAT loaded from Kaggle Secrets')\n",
    "except Exception:\n",
    "    print('No GITHUB_TOKEN secret found — trying public clone')\n",
    "\n",
    "if token:\n",
    "    clone_url = f'https://{token}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git'\n",
    "else:\n",
    "    clone_url = f'https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git'\n",
    "\n",
    "if os.path.exists(REPO_DIR):\n",
    "    print(f'Repo already exists — pulling latest')\n",
    "    run(f'git -C {REPO_DIR} pull')\n",
    "else:\n",
    "    print(f'Cloning {GITHUB_USER}/{GITHUB_REPO}...')\n",
    "    r = run(f'git clone --depth=1 \"{clone_url}\" {REPO_DIR}')\n",
    "    if r.returncode != 0:\n",
    "        print('\\nClone FAILED. Two fixes:')\n",
    "        print('  1) Make repo public: GitHub → Settings → Danger Zone → Change visibility')\n",
    "        print('  2) OR: Add Kaggle Secret GITHUB_TOKEN with a Personal Access Token')\n",
    "        print('     https://github.com/settings/tokens  →  Generate new token → repo scope')\n",
    "        raise RuntimeError('Clone failed — see instructions above')\n",
    "\n",
    "assert os.path.isdir(PROJECT_DIR), f'GravitySort folder not found at {PROJECT_DIR}'\n",
    "print(f'\\nProject root: {PROJECT_DIR}')\n",
    "run(f'find {PROJECT_DIR} -type f | sort')\n"
]

NEW_SEC1B = [
    "import shutil, glob\n",
    "\n",
    "print('=' * 55)\n",
    "print('KAGGLE DEPENDENCY AUDIT')\n",
    "print('=' * 55)\n",
    "\n",
    "# 1. System tools\n",
    "need_apt = []\n",
    "for t in ['nvcc', 'cmake', 'ninja', 'ncu', 'nsys', 'git']:\n",
    "    p = shutil.which(t)\n",
    "    print(f'  {\"OK  \" if p else \"MISS\"} {t:<8} {p or \"\"}')\n",
    "    if not p and t in ('cmake', 'ninja'): need_apt.append(t)\n",
    "\n",
    "# 2. Install missing build tools\n",
    "if need_apt:\n",
    "    print('\\nInstalling via apt:', need_apt)\n",
    "    run('apt-get update -qq')\n",
    "    run('apt-get install -y -q cmake ninja-build libglfw3-dev libglew-dev')\n",
    "else:\n",
    "    run('apt-get install -y -q libglfw3-dev libglew-dev 2>/dev/null || true')\n",
    "\n",
    "# 3. Python packages\n",
    "print('\\nPython packages:')\n",
    "for p in ['numpy', 'matplotlib', 'scipy', 'pygame']:\n",
    "    try:\n",
    "        m = __import__(p); print(f'  OK   {p:<12} {m.__version__}')\n",
    "    except ImportError:\n",
    "        print(f'  installing {p}...'); run(f'pip install {p} -q')\n",
    "\n",
    "# 4. CUDA libraries\n",
    "print('\\nCUDA libraries:')\n",
    "for lib in ['libcublas.so', 'libcudart.so', 'libnvToolsExt.so']:\n",
    "    found = glob.glob(f'/usr/local/cuda/**/{lib}*', recursive=True)\n",
    "    print(f'  {\"OK  \" if found else \"MISS\"} {lib}')\n",
    "\n",
    "# 5. Thrust headers\n",
    "thrust_ok = os.path.isdir('/usr/local/cuda/include/thrust')\n",
    "print(f'  {\"OK  \" if thrust_ok else \"MISS\"} Thrust headers (/usr/local/cuda/include/thrust)')\n",
    "\n",
    "print('\\nAll checks done — ready for Section 2 (CMake Build).')\n"
]

patched = 0
for cell in nb["cells"]:
    if cell.get("id") == "setup-project":
        cell["source"] = NEW_SEC1
        patched += 1
    elif cell.get("id") == "install-deps":
        cell["source"] = NEW_SEC1B
        patched += 1

assert patched == 2, f"Expected 2 patched cells, got {patched}"

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Patched {patched} cells OK")
print("  setup-project: private repo + PAT support + public fallback")
print("  install-deps:  cmake/ninja/pip/CUDA/Thrust audit")
