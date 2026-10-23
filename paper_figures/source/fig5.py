"""Functional sample accuracy versus design coverage on PICBench.

Hand-drawn framing with exact, unjittered data coordinates and equal scales.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from style import BLUE, CORAL, GRAY, MUTED, NAVY, TEAL, WHITE, save, setup

W, H = 7.0, 4.1
# name, functional Pass@1, functional Pass@5, fill, emphasized
MODELS = (
    ("Qwen3.5-9B", 0.83, 4.17, CORAL, True),
    ("DeepSeek-V4-Pro", 31.67, 62.50, TEAL, True),
    ("Qwen3.6-27B", 30.83, 70.83, BLUE, True),
    ("GLM-5.3", 39.17, 54.17, CORAL, True),
    ("DeepSeek-V4.1-Flash", 30.00, 58.33, WHITE, False),
)


def build() -> None:
    fig = plt.figure(figsize=(W, H), dpi=140)
    ax = fig.add_axes([0.10, 0.18, 0.52, 0.65])
    fig.text(0.035, 0.965, "Accuracy is not coverage", color=NAVY, fontsize=14,
             weight="bold", va="top")
    fig.text(0.035, 0.875, "Functional scores  /  24 designs × 5 samples", fontsize=9,
             color=MUTED)
    ax.plot([0, 80], [0, 80], color=GRAY, ls=(0, (4, 4)), lw=1, zorder=1)
    ax.text(57, 52, "Pass@5 = Pass@1", rotation=45, fontsize=7.5,
            color=MUTED, ha="center", va="center")
    # Name labels for the clustered API points sit outside the plotting frame.
    # Leaders stay outside the data cloud except for their short starting legs.
    destinations = {
        "Qwen3.6-27B": (87, 74),
        "DeepSeek-V4-Pro": (87, 62),
        "DeepSeek-V4.1-Flash": (87, 48),
        "GLM-5.3": (87, 32),
    }
    for name, p1, p5, fill, _ in MODELS:
        ax.scatter(p1, p5, s=82, marker="o" if fill != WHITE else "D",
                   facecolor=fill, edgecolor=NAVY, linewidth=0.85, zorder=5)
        if name == "Qwen3.5-9B":
            ax.annotate("Qwen3.5-9B", (p1, p5), xytext=(9, 11),
                        textcoords="offset points", fontsize=9, color=NAVY,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7})
            continue
        tx, ty = destinations[name]
        ax.annotate("", xy=(p1 + 1.5, p5), xytext=(tx - 2, ty),
                    arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.9,
                                "connectionstyle": "arc3,rad=0.06"},
                    annotation_clip=False, zorder=2)
        ax.text(tx, ty + 1.5, name, fontsize=9.4, color=NAVY,
                weight="bold" if name in ("GLM-5.3", "Qwen3.6-27B") else "normal",
                clip_on=False, va="bottom")
        ax.text(tx, ty - 2.2, f"{p1:.2f}% samples  /  {p5:.2f}% designs",
                fontsize=8, color=MUTED, clip_on=False, va="top")
    ax.set_xlim(-3, 80)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(labelsize=8, length=3)
    ax.grid(color=GRAY, alpha=0.28, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRAY)
    ax.set_xlabel("Functional Pass@1 (%)", fontsize=9.5, labelpad=6)
    ax.set_ylabel("Functional Pass@5 (%)", fontsize=9.5, labelpad=6)
    save(fig, "fig5_model_pareto")


if __name__ == "__main__":
    setup()
    build()
