#!/usr/bin/env python3
"""
GravitySort — Python Frontend (pybind11)
Sorting visualization using matplotlib/pygame animation.
"""

import numpy as np
import time
import sys
import os

# Try to import the compiled pybind11 module (only available after build)
try:
    import gravity_sort_py as gs
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    print("[INFO] Native CUDA module not available — using numpy simulation")

# ─── Numpy fallback simulation ────────────────────────────────────────────
def simulate_sort_steps(arr, algorithm="bitonic"):
    """Generator: yields array states during sort for animation."""
    a = arr.copy()
    n = len(a)
    if algorithm == "bitonic":
        k = 2
        while k <= n:
            j = k // 2
            while j >= 1:
                for i in range(n):
                    l = i ^ j
                    if l > i:
                        asc = (i & k) == 0
                        if (asc and a[i] > a[l]) or (not asc and a[i] < a[l]):
                            a[i], a[l] = a[l], a[i]
                yield a.copy()
                j //= 2
            k *= 2
    elif algorithm == "bubble":
        for _ in range(n):
            for j in range(n - 1):
                if a[j] > a[j+1]:
                    a[j], a[j+1] = a[j+1], a[j]
            yield a.copy()
    return a

# ─── Matplotlib bar animation ─────────────────────────────────────────────
def animate_sort_matplotlib(N=64, algorithm="bitonic", interval_ms=30):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("matplotlib not available — run: pip install matplotlib")
        return

    arr = np.random.permutation(N).astype(float)
    steps = list(simulate_sort_steps(arr, algorithm))
    if not steps:
        steps = [np.sort(arr)]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.set_title(f"GravitySort ⚡ {algorithm.title()} Sort  N={N}",
                 color="#f0f6fc", fontsize=13, fontweight="bold")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, N))
    bars = ax.bar(range(N), steps[0], color=colors, width=0.85, edgecolor="none")
    ax.set_ylim(0, N + 1)
    ax.set_xlim(-0.5, N - 0.5)

    step_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                        color="#8b949e", fontsize=9, va="top")

    def update(frame):
        data = steps[min(frame, len(steps) - 1)]
        for bar, h in zip(bars, data):
            bar.set_height(h)
        pct = int((frame + 1) / len(steps) * 100)
        step_text.set_text(f"Step {frame+1}/{len(steps)}  ({pct}%)")
        return bars

    ani = animation.FuncAnimation(fig, update, frames=len(steps),
                                  interval=interval_ms, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani

# ─── Pygame live visualization ────────────────────────────────────────────
def animate_sort_pygame(N=128, algorithm="bitonic"):
    try:
        import pygame
    except ImportError:
        print("pygame not available — run: pip install pygame")
        return

    arr = np.random.permutation(N).astype(float)
    steps = list(simulate_sort_steps(arr, algorithm))

    pygame.init()
    W, H = 900, 500
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"GravitySort ⚡ {algorithm.title()} Sort")
    clock = pygame.time.Clock()
    BG    = (13, 17, 23)
    BAR_W = max(1, W // N)

    for frame, state in enumerate(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return

        screen.fill(BG)
        for i, val in enumerate(state):
            ratio  = val / N
            color  = (int(255 * ratio), int(100 + 100 * (1-ratio)), 220)
            bar_h  = int(ratio * (H - 40))
            x      = i * BAR_W
            pygame.draw.rect(screen, color, (x, H - bar_h - 20, BAR_W - 1, bar_h))

        pygame.display.flip()
        clock.tick(60)

    # Hold final frame
    running = True
    while running:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                running = False
    pygame.quit()

# ─── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GravitySort Python Visualization")
    parser.add_argument("--n",         type=int,  default=64,       help="Array size")
    parser.add_argument("--algo",      type=str,  default="bitonic", choices=["bitonic","bubble"])
    parser.add_argument("--backend",   type=str,  default="matplotlib", choices=["matplotlib","pygame"])
    parser.add_argument("--interval",  type=int,  default=30,       help="Animation interval ms")
    args = parser.parse_args()

    print(f"GravitySort ⚡ Python Frontend")
    print(f"  Algorithm : {args.algo}  N={args.n}  Backend={args.backend}")
    if args.backend == "pygame":
        animate_sort_pygame(args.n, args.algo)
    else:
        animate_sort_matplotlib(args.n, args.algo, args.interval)
