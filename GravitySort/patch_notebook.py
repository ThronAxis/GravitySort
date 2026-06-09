"""
Patches GravitySort_Kaggle.ipynb:
  1. Fixes GitHub URL  →  ThronAxis/GravityShort_Project_file
  2. Fixes PROJECT_DIR →  REPO_DIR/GravitySort  (subfolder)
  3. Replaces Section 1b (install-deps) with a proper Kaggle audit cell
     that checks nvcc/cmake/ninja/ncu/nsys, installs missing tools via apt,
     installs missing Python packages via pip, and checks cublas/cudart.
"""
import json, sys, copy

NB_PATH = "GravitySort_Kaggle.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# ── New Section 1 (setup-project) source ─────────────────────────────────
NEW_SEC1 = [
    "import os\n",
    "\n",
    "# ── GitHub repo (ThronAxis/GravityShort_Project_file) ──────────────────\n",
    "GITHUB_URL  = 'https://github.com/ThronAxis/GravityShort_Project_file.git'\n",
    "REPO_DIR    = '/kaggle/working/GravityShort_Project_file'\n",
    "PROJECT_DIR = f'{REPO_DIR}/GravitySort'   # GravitySort lives inside the repo\n",
    "\n",
    "if os.path.exists(REPO_DIR):\n",
    "    print(f'Repo already at {REPO_DIR} — pulling latest')\n",
    "    run(f'git -C {REPO_DIR} pull')\n",
    "else:\n",
    "    print('Cloning from GitHub...')\n",
    "    r = run(f'git clone --depth=1 {GITHUB_URL} {REPO_DIR}')\n",
    "    assert r.returncode == 0, 'Clone FAILED — make sure the repo is PUBLIC on GitHub'\n",
    "\n",
    "assert os.path.isdir(PROJECT_DIR), f'GravitySort not found at {PROJECT_DIR}'\n",
    "print(f'Project root: {PROJECT_DIR}')\n",
    "run(f'find {PROJECT_DIR} -type f | sort')\n"
]

# ── New Section 1b (install-deps) source ─────────────────────────────────
NEW_SEC1B = [
    "import shutil, glob\n",
    "\n",
    "print('=' * 55)\n",
    "print('KAGGLE DEPENDENCY AUDIT')\n",
    "print('=' * 55)\n",
    "\n",
    "# ── 1. System tools ───────────────────────────────────────────\n",
    "need_apt = []\n",
    "for t in ['nvcc', 'cmake', 'ninja', 'ncu', 'nsys', 'git']:\n",
    "    p = shutil.which(t)\n",
    "    print(f'  {\"OK  \" if p else \"MISS\"} {t:<8} {p or \"\"}')\n",
    "    if not p and t in ('cmake', 'ninja'): need_apt.append(t)\n",
    "\n",
    "# ── 2. Install missing build tools via apt ────────────────────\n",
    "if need_apt:\n",
    "    print('\\nInstalling via apt:', need_apt)\n",
    "    run('apt-get update -qq')\n",
    "    run('apt-get install -y -q cmake ninja-build libglfw3-dev libglew-dev')\n",
    "else:\n",
    "    run('apt-get install -y -q libglfw3-dev libglew-dev 2>/dev/null || true')\n",
    "\n",
    "# ── 3. Python packages ────────────────────────────────────────\n",
    "print('\\nPython packages:')\n",
    "for p in ['numpy', 'matplotlib', 'scipy', 'pygame']:\n",
    "    try:\n",
    "        m = __import__(p); print(f'  OK   {p:<12} {m.__version__}')\n",
    "    except ImportError:\n",
    "        print(f'  installing {p}...'); run(f'pip install {p} -q')\n",
    "\n",
    "# ── 4. CUDA libraries ─────────────────────────────────────────\n",
    "print('\\nCUDA libraries:')\n",
    "for lib in ['libcublas.so', 'libcudart.so', 'libnvToolsExt.so']:\n",
    "    found = glob.glob(f'/usr/local/cuda/**/{lib}*', recursive=True)\n",
    "    print(f'  {\"OK  \" if found else \"MISS\"} {lib}')\n",
    "\n",
    "# ── 5. Thrust (header-only, ships with CUDA) ──────────────────\n",
    "thrust_ok = os.path.isdir('/usr/local/cuda/include/thrust')\n",
    "print(f'  {\"OK  \" if thrust_ok else \"MISS\"} Thrust headers')\n",
    "\n",
    "print('\\nAll checks done — ready for Section 2 (Build).')\n"
]

# ── Patch cells ───────────────────────────────────────────────────────────
patched = 0
for cell in nb["cells"]:
    if cell.get("id") == "setup-project":
        cell["source"] = NEW_SEC1
        patched += 1
    elif cell.get("id") == "install-deps":
        cell["source"] = NEW_SEC1B
        patched += 1

if patched != 2:
    print(f"WARNING: Expected to patch 2 cells, patched {patched}")
    sys.exit(1)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Patched {patched} cells in {NB_PATH}")
print("  setup-project  -> correct GitHub URL + repo subfolder path")
print("  install-deps   -> full Kaggle audit (tools/pip/cuda/thrust)")
