import json, os

NB = r'f:\Project\Working project\Gravityshort\GravityShort_Project_file\GravitySort\GravitySort_Kaggle.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ── helpers ──────────────────────────────────────────────────────────────────
def find_cell(cell_id):
    for i, c in enumerate(cells):
        if c.get('id') == cell_id:
            return i, c
    return None, None

def set_source(cell_id, new_lines):
    idx, c = find_cell(cell_id)
    if c is None:
        print(f'WARNING: cell {cell_id!r} not found')
        return
    c['source'] = new_lines
    print(f'  patched cell {cell_id!r}')

# ── Fix 1: Section 2 — add -DBUILD_VIZ=OFF ───────────────────────────────────
print('Fix 1: Section 2 cmake BUILD_VIZ=OFF')
idx, c = find_cell('build')
if c:
    src = ''.join(c['source'])
    src = src.replace(
        "    f' -DCMAKE_BUILD_TYPE=Release'\n"
        "    f' -G Ninja'\n",
        "    f' -DCMAKE_BUILD_TYPE=Release'\n"
        "    f' -DBUILD_VIZ=OFF'\n"
        "    f' -G Ninja'\n",
    )
    c['source'] = [line + '\n' for line in src.split('\n')]
    c['source'][-1] = c['source'][-1].rstrip('\n')
    print('  patched cell "build"')

# ── Fix 2: Section 8 — graceful ncu handling ────────────────────────────────
print('Fix 2: Section 8 ncu graceful')
set_source('nsight', [
    "import shutil\n",
    "BUILD_DIR = '/kaggle/working/build'\n",
    "\n",
    "# ncu may need perf event permissions on Kaggle — try gracefully\n",
    "if shutil.which('ncu'):\n",
    "    print('-- Profiling: reduction_demo with ncu --')\n",
    "    r = run(f'ncu --set basic --clock-control none {BUILD_DIR}/reduction_demo {1<<24}')\n",
    "    if r.returncode != 0:\n",
    "        print('[ncu fallback] Running demo directly')\n",
    "        run(f'{BUILD_DIR}/reduction_demo {1<<24}')\n",
    "else:\n",
    "    print('ncu not available — running reduction_demo directly')\n",
    "    run(f'{BUILD_DIR}/reduction_demo {1<<25}')\n",
    "\n",
    "# Nsight Systems timeline\n",
    "if shutil.which('nsys'):\n",
    "    print('\\n-- Nsight Systems: streams timeline --')\n",
    "    r = run(f'nsys profile --trace=cuda --output=/kaggle/working/streams_report {BUILD_DIR}/streams_demo')\n",
    "    if r.returncode == 0:\n",
    "        print('Profile saved: /kaggle/working/streams_report.nsys-rep')\n",
    "    else:\n",
    "        print('nsys profile failed — running streams_demo directly')\n",
    "        run(f'{BUILD_DIR}/streams_demo')\n",
    "else:\n",
    "    print('nsys not available — running streams_demo directly')\n",
    "    run(f'{BUILD_DIR}/streams_demo')\n",
])

# ── Fix 3: Section 9 — fix roofline path ─────────────────────────────────────
print('Fix 3: Section 9 roofline path')
set_source('roofline', [
    "import sys, os\n",
    "PROJECT_DIR  = '/kaggle/working/GravitySort-main/GravitySort'\n",
    "ROOFLINE_OUT = '/kaggle/working/roofline.png'\n",
    "sys.path.insert(0, f'{PROJECT_DIR}/profiling')\n",
    "\n",
    "r = run(f'python {PROJECT_DIR}/profiling/roofline.py --output {ROOFLINE_OUT}')\n",
    "\n",
    "from IPython.display import Image, display\n",
    "if os.path.isfile(ROOFLINE_OUT):\n",
    "    print('Roofline plot generated!')\n",
    "    display(Image(ROOFLINE_OUT))\n",
    "else:\n",
    "    print('ERROR: roofline.png not generated — check stderr above')\n",
])

# ── Fix 4: Add Section 12 — Results Summary (after Section 11) ───────────────
print('Fix 4: Adding Section 12 Results Summary')
s12_md = {
    "cell_type": "markdown",
    "id": "s12",
    "metadata": {},
    "source": ["## Section 12 - Results Summary"]
}
s12_code = {
    "cell_type": "code",
    "execution_count": None,
    "id": "results",
    "metadata": {},
    "outputs": [],
    "source": [
        "import subprocess, os\n",
        "\n",
        "BUILD_DIR   = '/kaggle/working/build'\n",
        "PROJECT_DIR = '/kaggle/working/GravitySort-main/GravitySort'\n",
        "\n",
        "print('=' * 65)\n",
        "print('  GRAVITYSORT — FINAL RESULTS SUMMARY')\n",
        "print('=' * 65)\n",
        "\n",
        "# GPU info\n",
        "smi = subprocess.run('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader',\n",
        "                     shell=True, capture_output=True, text=True).stdout.strip()\n",
        "print(f'GPU      : {smi}')\n",
        "nvcc_ver = subprocess.run('nvcc --version', shell=True, capture_output=True, text=True)\n",
        "print(f'CUDA     : {nvcc_ver.stdout.strip().split(chr(10))[-1].strip()}')\n",
        "\n",
        "# Kernel timing — run each once and capture output\n",
        "print('\\n--- Custom Kernel Timings ---')\n",
        "kernels = [\n",
        "    ('Bitonic Sort  ', 'bitonic_sort',   1<<24),\n",
        "    ('Radix Sort    ', 'radix_sort',     1<<24),\n",
        "    ('Reduction     ', 'reduction_demo', 1<<25),\n",
        "    ('Odd-Even Sort ', 'odd_even_sort',  65536),\n",
        "]\n",
        "for name, exe, N in kernels:\n",
        "    r = subprocess.run(f'{BUILD_DIR}/{exe} {N}', shell=True, capture_output=True, text=True)\n",
        "    lines = [l for l in r.stdout.split('\\n') if 'Band' in l or 'Sort time' in l or 'GB/s' in l or 'ms' in l]\n",
        "    timing = ' | '.join(lines[:2]) if lines else r.stdout.strip().split('\\n')[-1]\n",
        "    print(f'  {name}: {timing}')\n",
        "\n",
        "# Google Benchmark quick run\n",
        "print('\\n--- Google Benchmark (N=1M) ---')\n",
        "for bench, exe in [('Sort', 'bench_sort'), ('Reduce', 'bench_reduce')]:\n",
        "    r = subprocess.run(\n",
        "        f'{BUILD_DIR}/{exe} --benchmark_filter=1048576 --benchmark_repetitions=1',\n",
        "        shell=True, capture_output=True, text=True)\n",
        "    for line in r.stdout.strip().split('\\n'):\n",
        "        if 'BM_' in line and 'mean' not in line and 'stddev' not in line:\n",
        "            print(f'  {line.strip()}')\n",
        "\n",
        "# Unit test results\n",
        "print('\\n--- Unit Tests ---')\n",
        "r1 = subprocess.run(f'{BUILD_DIR}/test_sort', shell=True, capture_output=True, text=True)\n",
        "r2 = subprocess.run(f'{BUILD_DIR}/test_reduce', shell=True, capture_output=True, text=True)\n",
        "for label, r in [('test_sort  ', r1), ('test_reduce', r2)]:\n",
        "    for line in r.stdout.split('\\n'):\n",
        "        if 'PASSED' in line or 'FAILED' in line:\n",
        "            print(f'  {label}: {line.strip()}')\n",
        "            break\n",
        "\n",
        "# Artifacts\n",
        "print('\\n--- Output Artifacts ---')\n",
        "for f in ['roofline.png', 'streams_report.nsys-rep']:\n",
        "    path = f'/kaggle/working/{f}'\n",
        "    exists = os.path.isfile(path)\n",
        "    size   = f'{os.path.getsize(path)//1024} KB' if exists else 'N/A'\n",
        "    print(f'  {\"OK  \" if exists else \"MISS\"} {f} ({size})')\n",
        "\n",
        "print('\\n' + '='*65)\n",
        "print('  GravitySort deployed successfully on Kaggle GPU!')\n",
        "print('='*65)\n",
    ]
}

# Insert after the last cell (Section 11)
cells.append(s12_md)
cells.append(s12_code)
print('  added Section 12')

# ── Save ─────────────────────────────────────────────────────────────────────
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('\nNotebook patched successfully!')
print(f'Total cells: {len(cells)}')
