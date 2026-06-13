"""
Patches ONLY the build cell (Section 2) with better diagnostics and error output.
"""
import json

NB_PATH = "GravitySort/GravitySort_Kaggle.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

NEW_BUILD = [
    "import subprocess, os\n",
    "\n",
    "PROJECT_DIR = '/kaggle/working/GravitySort-main'\n",
    "BUILD_DIR   = '/kaggle/working/build'\n",
    "os.makedirs(BUILD_DIR, exist_ok=True)\n",
    "\n",
    "# ── Diagnostics first ────────────────────────────────────────\n",
    "print('== Pre-build checks ==')\n",
    "print('PROJECT_DIR exists:', os.path.isdir(PROJECT_DIR))\n",
    "print('CMakeLists.txt exists:', os.path.isfile(f'{PROJECT_DIR}/CMakeLists.txt'))\n",
    "r = subprocess.run('nvcc --version', shell=True, capture_output=True, text=True)\n",
    "print('nvcc:', r.stdout.strip().split('\\n')[-1] if r.returncode==0 else 'NOT FOUND')\n",
    "r = subprocess.run('cmake --version', shell=True, capture_output=True, text=True)\n",
    "print('cmake:', r.stdout.strip().split('\\n')[0] if r.returncode==0 else 'NOT FOUND')\n",
    "r = subprocess.run('ninja --version', shell=True, capture_output=True, text=True)\n",
    "print('ninja:', r.stdout.strip() if r.returncode==0 else 'NOT FOUND - installing...')\n",
    "if r.returncode != 0:\n",
    "    subprocess.run('apt-get install -y -q ninja-build', shell=True)\n",
    "\n",
    "# ── GPU arch ─────────────────────────────────────────────────\n",
    "smi = subprocess.run('nvidia-smi --query-gpu=name --format=csv,noheader',\n",
    "                     shell=True, capture_output=True, text=True).stdout.strip()\n",
    "print(f'\\nGPU: {smi}')\n",
    "arch_map = {'T4':'75','V100':'70','P100':'60','A100':'80','A6000':'86','RTX 30':'86','RTX 40':'89'}\n",
    "arch = next((v for k,v in arch_map.items() if k.lower() in smi.lower()), '75')\n",
    "print(f'CUDA arch: sm_{arch}')\n",
    "\n",
    "# ── CMake configure ───────────────────────────────────────────\n",
    "print('\\n== CMake Configure ==')\n",
    "cmd = (\n",
    "    f'cmake -S {PROJECT_DIR} -B {BUILD_DIR}'\n",
    "    f' -DCMAKE_CUDA_ARCHITECTURES={arch}'\n",
    "    f' -DCMAKE_BUILD_TYPE=Release'\n",
    "    f' -G Ninja'\n",
    ")\n",
    "print('CMD:', cmd)\n",
    "r = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n",
    "print(r.stdout[-3000:] if r.stdout else '')\n",
    "if r.stderr: print('STDERR:', r.stderr[-2000:])\n",
    "if r.returncode != 0:\n",
    "    raise RuntimeError(f'CMake configure failed (exit {r.returncode})')\n",
    "\n",
    "# ── Build ─────────────────────────────────────────────────────\n",
    "ncpu = os.cpu_count() or 4\n",
    "print(f'\\n== CMake Build (parallel={ncpu}) ==')\n",
    "r = subprocess.run(f'cmake --build {BUILD_DIR} --parallel {ncpu}',\n",
    "                   shell=True, capture_output=True, text=True)\n",
    "print(r.stdout[-4000:] if r.stdout else '')\n",
    "if r.stderr: print('STDERR:', r.stderr[-2000:])\n",
    "if r.returncode != 0:\n",
    "    raise RuntimeError(f'Build failed (exit {r.returncode})')\n",
    "\n",
    "print('\\nBuild complete! Executables:')\n",
    "r = subprocess.run(f'ls -lh {BUILD_DIR}/', shell=True, capture_output=True, text=True)\n",
    "print(r.stdout)\n",
]

patched = 0
for cell in nb["cells"]:
    if cell.get("id") == "build":
        cell["source"] = NEW_BUILD
        patched += 1

assert patched == 1, f"Expected 1 patched cell, got {patched}"

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Patched build cell with full diagnostics.")
