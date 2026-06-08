#!/usr/bin/env python3
"""
GravitySort — Roofline Model Plot
───────────────────────────────────────────────────────────────────────────────
Generates a roofline model chart for Kaggle T4 / A100 GPUs, overlays
actual GravitySort kernel measurements, and saves as roofline.png for
embedding in README.md.

Usage:
    python profiling/roofline.py [--results results.json]

results.json format (populate after running benchmarks):
{
  "kernels": [
    {"name": "Bitonic Sort",      "gflops": 45.2,  "intensity": 0.25},
    {"name": "Radix Sort",        "gflops": 210.5, "intensity": 1.8},
    {"name": "Reduction (naive)", "gflops": 12.1,  "intensity": 0.05},
    {"name": "Reduction (warp)",  "gflops": 890.0, "intensity": 0.50},
    {"name": "Reduction (vec4)",  "gflops": 1250.0,"intensity": 0.50}
  ]
}
"""

import json, argparse, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── GPU hardware limits ───────────────────────────────────────────────────
GPU_PROFILES = {
    "T4  (Kaggle free)": {
        "peak_tflops":  8.1,    # FP32 TFLOPS
        "peak_bw_tbps": 0.320,  # memory bandwidth (320 GB/s)
        "color": "#4fc3f7",
    },
    "A100 (Kaggle P100≈)": {
        "peak_tflops":  19.5,
        "peak_bw_tbps": 0.900,
        "color": "#81c784",
    },
}

# ─── Default demo kernel measurements (replace with actual ncu output) ─────
DEFAULT_KERNELS = [
    {"name": "Bitonic Sort",        "gflops":   42.0,   "intensity": 0.25,  "marker": "o"},
    {"name": "Radix Sort",          "gflops":  195.0,   "intensity": 1.60,  "marker": "s"},
    {"name": "Odd-Even Sort",       "gflops":   18.0,   "intensity": 0.10,  "marker": "^"},
    {"name": "Reduction (naive)",   "gflops":   10.0,   "intensity": 0.04,  "marker": "D"},
    {"name": "Reduction (shared)",  "gflops":  420.0,   "intensity": 0.35,  "marker": "D"},
    {"name": "Reduction (warp)",    "gflops":  810.0,   "intensity": 0.48,  "marker": "D"},
    {"name": "Reduction (vec4)",    "gflops": 1180.0,   "intensity": 0.50,  "marker": "D"},
]

KERNEL_COLORS = [
    "#ef5350", "#ab47bc", "#42a5f5", "#26a69a",
    "#ffca28", "#ff7043", "#78909c"
]

def roofline_peak(intensity, peak_tflops, peak_bw_tbps):
    """Returns achievable GFLOPS at given arithmetic intensity (FLOP/byte)."""
    compute_bound = peak_tflops * 1000  # GFLOPS
    memory_bound  = intensity * peak_bw_tbps * 1000  # GFLOPS
    return min(compute_bound, memory_bound)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None, help="JSON file with kernel measurements")
    parser.add_argument("--output",  default="profiling/roofline.png")
    args = parser.parse_args()

    kernels = DEFAULT_KERNELS
    if args.results:
        with open(args.results) as f:
            kernels = json.load(f)["kernels"]

    # ─── Figure setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    intensities = np.logspace(-2, 2, 500)

    # ─── Draw rooflines for each GPU ───────────────────────────────────────
    legend_handles = []
    for gpu_name, gpu in GPU_PROFILES.items():
        roof = [roofline_peak(i, gpu["peak_tflops"], gpu["peak_bw_tbps"])
                for i in intensities]
        line, = ax.loglog(intensities, roof,
                          color=gpu["color"], linewidth=2.5, linestyle="--",
                          label=f"{gpu_name}  (Peak: {gpu['peak_tflops']} TFLOPS, "
                                f"{int(gpu['peak_bw_tbps']*1000)} GB/s)")
        # Ridge point
        ridge = gpu["peak_tflops"] / gpu["peak_bw_tbps"]
        ax.axvline(ridge, color=gpu["color"], alpha=0.25, linewidth=1)
        ax.text(ridge * 1.05,
                gpu["peak_tflops"] * 1000 * 0.7,
                f"Ridge\n{ridge:.1f}",
                color=gpu["color"], fontsize=8, alpha=0.8)
        legend_handles.append(line)

    # ─── Plot kernels ───────────────────────────────────────────────────────
    for i, k in enumerate(kernels):
        color = KERNEL_COLORS[i % len(KERNEL_COLORS)]
        scatter = ax.scatter(k["intensity"], k["gflops"],
                             color=color, s=120, zorder=5,
                             marker=k.get("marker", "o"),
                             edgecolors="white", linewidths=0.8)
        ax.annotate(k["name"],
                    (k["intensity"], k["gflops"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8.5, color=color, fontweight="bold")
        legend_handles.append(
            mpatches.Patch(color=color, label=f"{k['name']}  ({k['gflops']:.0f} GFLOPS)"))

    # ─── Labels & grid ─────────────────────────────────────────────────────
    ax.set_xlabel("Arithmetic Intensity  [FLOP / byte]",
                  color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Performance  [GFLOPS]",
                  color="#c9d1d9", fontsize=11)
    ax.set_title("GravitySort — Roofline Model\n"
                 "Kaggle GPU Environment  (T4 / A100)",
                 color="#f0f6fc", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="#8b949e")
    ax.grid(True, which="both", color="#21262d", linewidth=0.8, linestyle=":")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1, 3e4)

    legend = ax.legend(handles=legend_handles, loc="upper left",
                       facecolor="#161b22", edgecolor="#30363d",
                       labelcolor="#c9d1d9", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✓ Roofline model saved → {args.output}")

if __name__ == "__main__":
    main()
