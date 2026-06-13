"""
Completely rewrites GravitySort_Kaggle.ipynb with a clean, working version.
Uses wget zip download (avoids git DNS issues on Kaggle).
Repo: ThronAxis/GravitySort (renamed from GravityShort_Project_file)
"""
import json

NB = {
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"},
  "kaggle": {
   "accelerator": "gpu",
   "isGpuEnabled": True,
   "isInternetEnabled": True,
   "language": "python",
   "sourceType": "notebook"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5,
 "cells": []
}

def md(id_, text):
    return {"cell_type": "markdown", "id": id_, "metadata": {}, "source": [text]}

def code(id_, lines):
    return {"cell_type": "code", "execution_count": None, "id": id_,
            "metadata": {}, "outputs": [], "source": lines}

# ── Section 0: GPU Check ─────────────────────────────────────────────────────
NB["cells"].append(md("hdr", "# GravitySort - GPU Sorting & ML Reduction Framework\n\n**Kaggle GPU Notebook** | Builds and benchmarks CUDA C++ kernels on T4/P100."))

NB["cells"].append(md("s0", "## Section 0 - GPU & Environment Check"))
NB["cells"].append(code("gpu-check", [
    "import subprocess, os, sys\n",
    "\n",
    "def run(cmd):\n",
    "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n",
    "    out = (r.stdout + r.stderr).strip()\n",
    "    if out: print(out)\n",
    "    return r\n",
    "\n",
    "print('='*60)\n",
    "print('GPU INFO')\n",
    "print('='*60)\n",
    "run('nvidia-smi')\n",
    "run('nvcc --version')\n",
    "run('cmake --version')\n",
    "run('ninja --version')\n",
    "print('CPUs:', os.cpu_count())\n",
]))

# ── Section 1: Download Project ───────────────────────────────────────────────
NB["cells"].append(md("s1", "## Section 1 - Download Project"))
NB["cells"].append(code("download", [
    "import os\n",
    "\n",
    "ZIP_URL     = 'https://github.com/ThronAxis/GravitySort/archive/refs/heads/main.zip'\n",
    "ZIP_PATH    = '/kaggle/working/GravitySort.zip'\n",
    "PROJECT_DIR = '/kaggle/working/GravitySort-main'\n",
    "\n",
    "if os.path.isdir(PROJECT_DIR):\n",
    "    print('Already downloaded:', PROJECT_DIR)\n",
    "else:\n",
    "    print('Downloading GravitySort from GitHub...')\n",
    "    r = run(f'wget -q --show-progress -O {ZIP_PATH} \"{ZIP_URL}\"')\n",
    "    if r.returncode != 0:\n",
    "        print('wget failed, trying curl...')\n",
    "        r = run(f'curl -L -o {ZIP_PATH} \"{ZIP_URL}\"')\n",
    "    assert r.returncode == 0, 'Download failed - check Internet is ON in Settings'\n",
    "    print('Extracting...')\n",
    "    run(f'unzip -q {ZIP_PATH} -d /kaggle/working/')\n",
    "\n",
    "assert os.path.isdir(PROJECT_DIR), f'Not found: {PROJECT_DIR}'\n",
    "print('Project files:')\n",
    "run(f'find {PROJECT_DIR} -type f | sort')\n",
]))

# ── Section 1b: Dependencies ───────────────────────────────────────────────────
NB["cells"].append(md("s1b", "## Section 1b - Install Dependencies"))
NB["cells"].append(code("deps", [
    "import shutil, glob\n",
    "\n",
    "print('='*55)\n",
    "print('DEPENDENCY AUDIT')\n",
    "print('='*55)\n",
    "\n",
    "# System tools\n",
    "need_apt = []\n",
    "for t in ['nvcc','cmake','ninja','ncu','nsys','git']:\n",
    "    p = shutil.which(t)\n",
    "    print(f'  {\"OK  \" if p else \"MISS\"} {t:<8} {p or \"\"}')\n",
    "    if not p and t in ('cmake','ninja'): need_apt.append(t)\n",
    "\n",
    "# Install missing tools\n",
    "if need_apt:\n",
    "    print('Installing:', need_apt)\n",
    "    run('apt-get update -qq')\n",
    "    run('apt-get install -y -q cmake ninja-build')\n",
    "\n",
    "run('apt-get install -y -q libglfw3-dev libglew-dev 2>/dev/null || true')\n",
    "\n",
    "# Python packages\n",
    "print('\\nPython packages:')\n",
    "for p in ['numpy','matplotlib','scipy','pygame']:\n",
    "    try:\n",
    "        m = __import__(p); print(f'  OK   {p} {m.__version__}')\n",
    "    except ImportError:\n",
    "        print(f'  Installing {p}...'); run(f'pip install {p} -q')\n",
    "\n",
    "# CUDA libs\n",
    "print('\\nCUDA libraries:')\n",
    "for lib in ['libcublas.so','libcudart.so']:\n",
    "    found = glob.glob(f'/usr/local/cuda/**/{lib}*', recursive=True)\n",
    "    print(f'  {\"OK  \" if found else \"MISS\"} {lib}')\n",
    "\n",
    "thrust_ok = os.path.isdir('/usr/local/cuda/include/thrust')\n",
    "print(f'  {\"OK  \" if thrust_ok else \"MISS\"} Thrust headers')\n",
    "print('\\nReady for build.')\n",
]))

# ── Section 2: CMake Build ─────────────────────────────────────────────────────
NB["cells"].append(md("s2", "## Section 2 - CMake Build (4-6 minutes)"))
NB["cells"].append(code("build", [
    "import subprocess, os\n",
    "\n",
    "PROJECT_DIR = '/kaggle/working/GravitySort-main'\n",
    "BUILD_DIR   = '/kaggle/working/build'\n",
    "os.makedirs(BUILD_DIR, exist_ok=True)\n",
    "\n",
    "# Detect GPU arch\n",
    "smi = subprocess.run('nvidia-smi --query-gpu=name --format=csv,noheader',\n",
    "                     shell=True, capture_output=True, text=True).stdout.strip()\n",
    "print(f'GPU: {smi}')\n",
    "arch_map = {'T4':'75','V100':'70','P100':'60','A100':'80','A6000':'86','RTX 30':'86','RTX 40':'89'}\n",
    "arch = next((v for k,v in arch_map.items() if k.lower() in smi.lower()), '75')\n",
    "print(f'CUDA arch: sm_{arch}')\n",
    "\n",
    "# Configure\n",
    "print('\\n-- CMake configure --')\n",
    "r = run(\n",
    "    f'cmake -S {PROJECT_DIR} -B {BUILD_DIR}'\n",
    "    f' -DCMAKE_CUDA_ARCHITECTURES={arch}'\n",
    "    f' -DCMAKE_BUILD_TYPE=Release'\n",
    "    f' -G Ninja'\n",
    ")\n",
    "if r.returncode != 0:\n",
    "    raise RuntimeError('CMake configure failed')\n",
    "\n",
    "# Build\n",
    "ncpu = os.cpu_count() or 4\n",
    "print(f'\\n-- Build (parallel={ncpu}) --')\n",
    "r = run(f'cmake --build {BUILD_DIR} --parallel {ncpu}')\n",
    "if r.returncode != 0:\n",
    "    raise RuntimeError('Build failed')\n",
    "\n",
    "print('\\nBuild complete. Executables:')\n",
    "run(f'ls -lh {BUILD_DIR}/ | grep -v CMake | grep -v Makefile')\n",
]))

# ── Section 3: Sorting Kernels ─────────────────────────────────────────────────
NB["cells"].append(md("s3", "## Section 3 - Sorting Kernels"))
NB["cells"].append(code("sort", [
    "BUILD_DIR = '/kaggle/working/build'\n",
    "\n",
    "for algo, exe in [('BITONIC','bitonic_sort'),('RADIX','radix_sort'),('ODD-EVEN','odd_even_sort')]:\n",
    "    print(f'\\n{\"=\"*50}')\n",
    "    print(f'{algo} SORT')\n",
    "    print('='*50)\n",
    "    sizes = [1<<20, 1<<24, 1<<26] if algo != 'ODD-EVEN' else [65536]\n",
    "    for N in sizes:\n",
    "        print(f'N={N:,}')\n",
    "        run(f'{BUILD_DIR}/{exe} {N}')\n",
]))

# ── Section 4: Reduction ───────────────────────────────────────────────────────
NB["cells"].append(md("s4", "## Section 4 - ML Reduction Kernels"))
NB["cells"].append(code("reduce", [
    "BUILD_DIR = '/kaggle/working/build'\n",
    "print('='*50)\n",
    "print('REDUCTION: 4 variants vs thrust::reduce')\n",
    "print('='*50)\n",
    "for N in [1<<24, 1<<25, 1<<27]:\n",
    "    print(f'\\nN={N:,}')\n",
    "    run(f'{BUILD_DIR}/reduction_demo {N}')\n",
]))

# ── Section 5: Memory ──────────────────────────────────────────────────────────
NB["cells"].append(md("s5", "## Section 5 - Memory Optimization Demos"))
NB["cells"].append(code("mem", [
    "BUILD_DIR = '/kaggle/working/build'\n",
    "print('-- Bank Conflict Demo --')\n",
    "run(f'{BUILD_DIR}/shared_mem_demo')\n",
    "print('\\n-- Stream Concurrency Demo --')\n",
    "run(f'{BUILD_DIR}/streams_demo')\n",
]))

# ── Section 6: Tensor ──────────────────────────────────────────────────────────
NB["cells"].append(md("s6", "## Section 6 - GravityTensor Operations"))
NB["cells"].append(code("tensor", [
    "BUILD_DIR = '/kaggle/working/build'\n",
    "run(f'{BUILD_DIR}/tensor_demo')\n",
]))

# ── Section 7: Benchmark ───────────────────────────────────────────────────────
NB["cells"].append(md("s7", "## Section 7 - Google Benchmark"))
NB["cells"].append(code("bench", [
    "BUILD_DIR = '/kaggle/working/build'\n",
    "print('-- Sort Benchmark --')\n",
    "run(f'{BUILD_DIR}/bench_sort --benchmark_format=console --benchmark_repetitions=3')\n",
    "print('\\n-- Reduce Benchmark --')\n",
    "run(f'{BUILD_DIR}/bench_reduce --benchmark_format=console --benchmark_repetitions=3')\n",
]))

# ── Section 8: Nsight ──────────────────────────────────────────────────────────
NB["cells"].append(md("s8", "## Section 8 - Nsight Profiling"))
NB["cells"].append(code("nsight", [
    "BUILD_DIR = '/kaggle/working/build'\n",
    "NCU = 'ncu --set full --clock-control none --target-processes all'\n",
    "print('-- Profiling: bitonic_sort --')\n",
    "run(f'{NCU} {BUILD_DIR}/bitonic_sort 4194304')\n",
    "print('\\n-- Nsight Systems: streams --')\n",
    "run(f'nsys profile --trace=cuda,nvtx --output=/kaggle/working/streams_report {BUILD_DIR}/streams_demo')\n",
    "print('Profile saved: /kaggle/working/streams_report.nsys-rep')\n",
]))

# ── Section 9: Roofline ────────────────────────────────────────────────────────
NB["cells"].append(md("s9", "## Section 9 - Roofline Model"))
NB["cells"].append(code("roofline", [
    "import sys\n",
    "PROJECT_DIR = '/kaggle/working/GravitySort-main'\n",
    "sys.path.insert(0, f'{PROJECT_DIR}/profiling')\n",
    "run(f'python {PROJECT_DIR}/profiling/roofline.py --output /kaggle/working/roofline.png')\n",
    "from IPython.display import Image, display\n",
    "display(Image('/kaggle/working/roofline.png'))\n",
]))

# ── Section 10: Visualization ──────────────────────────────────────────────────
NB["cells"].append(md("s10", "## Section 10 - Sort Animation"))
NB["cells"].append(code("viz", [
    "import numpy as np\n",
    "import matplotlib\n",
    "matplotlib.use('Agg')\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.animation as animation\n",
    "from IPython.display import HTML\n",
    "\n",
    "N = 64\n",
    "arr = np.random.permutation(N).astype(float)\n",
    "\n",
    "def bitonic_steps(a):\n",
    "    steps = [a.copy()]; n = len(a); k = 2\n",
    "    while k <= n:\n",
    "        j = k // 2\n",
    "        while j >= 1:\n",
    "            for i in range(n):\n",
    "                l = i ^ j\n",
    "                if l > i:\n",
    "                    asc = (i & k) == 0\n",
    "                    if (asc and a[i] > a[l]) or (not asc and a[i] < a[l]):\n",
    "                        a[i], a[l] = a[l], a[i]\n",
    "            steps.append(a.copy()); j //= 2\n",
    "        k *= 2\n",
    "    return steps\n",
    "\n",
    "steps = bitonic_steps(arr.copy())\n",
    "steps = steps[::max(1, len(steps)//60)]\n",
    "\n",
    "matplotlib.use('module://matplotlib_inline.backend_inline')\n",
    "import matplotlib.pyplot as plt\n",
    "fig, ax = plt.subplots(figsize=(12, 4))\n",
    "fig.patch.set_facecolor('#0d1117')\n",
    "ax.set_facecolor('#161b22')\n",
    "for spine in ax.spines.values(): spine.set_color('#30363d')\n",
    "ax.tick_params(colors='#8b949e')\n",
    "ax.set_title(f'GravitySort Bitonic Sort  N={N}', color='#f0f6fc', fontsize=13, fontweight='bold')\n",
    "colors = plt.cm.plasma(np.linspace(0.1, 0.95, N))\n",
    "bars = ax.bar(range(N), steps[0], color=colors, width=0.85, edgecolor='none')\n",
    "ax.set_ylim(0, N+2)\n",
    "info = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='#8b949e', fontsize=9, va='top')\n",
    "\n",
    "def update(frame):\n",
    "    data = steps[frame]\n",
    "    sc = plt.cm.plasma(data / N)\n",
    "    for bar, h, c in zip(bars, data, sc):\n",
    "        bar.set_height(h); bar.set_color(c)\n",
    "    info.set_text(f'Step {frame+1}/{len(steps)}')\n",
    "    return bars\n",
    "\n",
    "ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=80, blit=False)\n",
    "plt.tight_layout()\n",
    "HTML(ani.to_jshtml())\n",
]))

# ── Section 11: Tests ──────────────────────────────────────────────────────────
NB["cells"].append(md("s11", "## Section 11 - Unit Tests"))
NB["cells"].append(code("tests", [
    "BUILD_DIR = '/kaggle/working/build'\n",
    "print('='*50)\n",
    "print('UNIT TESTS')\n",
    "print('='*50)\n",
    "r1 = run(f'{BUILD_DIR}/test_sort   --gtest_color=yes')\n",
    "r2 = run(f'{BUILD_DIR}/test_reduce --gtest_color=yes')\n",
    "print()\n",
    "if r1.returncode == 0 and r2.returncode == 0:\n",
    "    print('ALL TESTS PASSED')\n",
    "else:\n",
    "    print('SOME TESTS FAILED - check output above')\n",
]))

NB_PATH = "GravitySort/GravitySort_Kaggle.ipynb"
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)

print(f"Written {len(NB['cells'])} cells to {NB_PATH}")
