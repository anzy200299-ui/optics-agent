"""Syntax versus function: paired rates on a shared, linear percent axis.

The plotted values are the five measured PICBench rows, not historical rows.
Hand-drawn strokes are decorative; coordinates retain their exact values.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from style import BLUE, BLUE_SOFT, CORAL, GRAY, MUTED, NAVY, TEAL, WHITE, save, setup

W, H = 7.0, 3.65
MODELS = (
    ("Qwen3.5-9B", 3.33, 0.83, CORAL, True),
    ("DeepSeek-V4-Pro", 62.50, 31.67, TEAL, True),
    ("Qwen3.6-27B", 63.33, 30.83, "none", False),
    ("GLM-5.3", 81.67, 39.17, BLUE, False),
    ("DeepSeek-V4.1-Flash", 53.33, 30.00, "none", False),
)


def build() -> None:
    fig, ax = plt.subplots(figsize=(W, H), dpi=140)
    fig.subplots_adjust(left=0.245, right=0.965, bottom=0.18, top=0.79)
    fig.text(0.035, 0.94, "A runnable netlist can still be wrong", fontsize=14,
             weight="bold", color=NAVY, va="top")
    fig.text(0.035, 0.845, "PICBench Without EF  /  24 designs × 5 samples", fontsize=9,
             color=MUTED)
    for i, (name, syntax, functional, _, _) in enumerate(MODELS):
        y = 4 - i
        ax.plot([0, 100], [y, y], color=GRAY, alpha=0.35, lw=0.7, zorder=0)
        ax.plot([functional, syntax], [y, y], color=BLUE_SOFT, lw=7,
                solid_capstyle="round", zorder=2)
        ax.scatter([syntax], [y], s=80, facecolors=WHITE, edgecolors=BLUE,
                   linewidths=1.5, zorder=4)
        ax.scatter([functional], [y], s=65, facecolors=TEAL, edgecolors=NAVY,
                   linewidths=0.7, zorder=5)
        # Even the two low Qwen rates are labelled separately, above and below.
        ax.annotate(f"{functional:.2f}", (functional, y), xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=8.5, color=TEAL, weight="bold")
        ax.annotate(f"{syntax:.2f}", (syntax, y), xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=BLUE, weight="bold")
    ax.set_yticks(range(4, -1, -1), [r[0] for r in MODELS], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(-2, 100)
    ax.set_ylim(-0.63, 4.68)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=8, length=3)
    ax.set_xlabel("Pass@1 (%)", fontsize=10, labelpad=7)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    handles = [
        plt.Line2D([], [], marker="o", ls="none", ms=7, markerfacecolor=TEAL,
                   markeredgecolor=NAVY, label="Spectrum match"),
        plt.Line2D([], [], marker="o", ls="none", ms=7, markerfacecolor=WHITE,
                   markeredgecolor=BLUE, label="Simulable"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.02, 1.25),
              frameon=False, ncol=2, fontsize=8.5, handletextpad=0.35,
              columnspacing=1.4)
    save(fig, "fig4_syntax_function_gap")


if __name__ == "__main__":
    setup()
    build()
