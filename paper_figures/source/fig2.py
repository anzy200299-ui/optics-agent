"""Circuit-first question generation, with a sparse hand-drawn workflow.

Manual allocation is not fitted to the revised failure audit. Eight SAX probe
wavelengths check finiteness, not equality to a target spectrum.
"""
from __future__ import annotations

from matplotlib.patches import Circle

from style import (
    AMBER, AMBER_SOFT, BLUE, BLUE_SOFT, CORAL, GRAY, GRAY_SOFT,
    LOCUS_COLOR, MUTED, NAVY, TEAL, TEAL_SOFT, WHITE,
    arrow, canvas, rounded, save, setup, text,
)

W, H = 7.0, 3.8
LOCI = (("Ellipsis", 280), ("Comment", 160), ("Wrong model", 280),
        ("Ports", 80), ("Functional", 120), ("Graft", 80))
# Public geometry used by check_geometry.py.
CW = (1.72, 1.72, 1.72)
XS = (0.12, 2.04, 3.96)
CARD_Y, CARD_H = 0.88, 2.30


def heading(ax, x, number, title, color):
    ax.add_patch(Circle((x + 0.18, 3.43), 0.125, facecolor=color,
                        edgecolor=NAVY, lw=0.7, zorder=3))
    text(ax, x + 0.18, 3.43, str(number), color=WHITE, ha="center",
         fontsize=9, fontweight="bold")
    text(ax, x + 0.39, 3.43, title, fontsize=10.5, fontweight="bold")


def graph(ax, x, y):
    # A schematic split--phase--merge circuit, not an evaluated example.
    for xx in (x + 0.16, x + 1.13):
        rounded(ax, xx, y + 0.11, 0.14, 0.53, BLUE_SOFT, lw=0.9, radius=0.04)
    for yy in (y + 0.22, y + 0.53):
        ax.plot([x + 0.30, x + 1.13], [yy, yy], color=NAVY, lw=1.5)
    ax.plot([x, x + 0.16], [y + 0.38, y + 0.38], color=NAVY, lw=1.5)
    ax.plot([x + 1.27, x + 1.44], [y + 0.38, y + 0.38], color=NAVY, lw=1.5)
    rounded(ax, x + 0.59, y + 0.47, 0.24, 0.12, AMBER, lw=0.7, radius=0.02, z=5)
    text(ax, x + 0.71, y + 0.77, "one parameter", fontsize=8,
         color=MUTED, ha="center")


def build() -> None:
    fig, ax = canvas(W, H)
    heading(ax, XS[0], 1, "Choose a locus", AMBER)
    heading(ax, XS[1], 2, "Build the graph", BLUE)
    heading(ax, XS[2], 3, "Check with SAX", TEAL)
    rounded(ax, XS[0], CARD_Y, CW[0], CARD_H, AMBER_SOFT, radius=0.08)
    rounded(ax, XS[1], CARD_Y, CW[1], CARD_H, BLUE_SOFT, radius=0.08)
    rounded(ax, XS[2], CARD_Y, CW[2], CARD_H, TEAL_SOFT, radius=0.08)
    for i, (name, count) in enumerate(LOCI):
        yy = 2.89 - i * 0.26
        ax.add_patch(Circle((XS[0] + 0.15, yy), 0.035,
                            facecolor=LOCUS_COLOR[name], edgecolor="none"))
        text(ax, XS[0] + 0.25, yy, name, fontsize=9)
        text(ax, XS[0] + CW[0] - 0.14, yy, str(count), ha="right",
             fontsize=9, fontweight="bold")
    text(ax, XS[0] + 0.14, 1.12, "1,000 questions · fixed allocation",
         fontsize=7.4, color=MUTED)
    graph(ax, XS[1] + 0.14, 2.03)
    text(ax, XS[1] + 0.86, 1.70, "Wire first. Describe later.",
         fontsize=9, ha="center", fontweight="bold")
    text(ax, XS[1] + 0.86, 1.40, "Widths {2, 3, 5, 6}", fontsize=8.5,
         ha="center", color=MUTED)
    text(ax, XS[1] + 0.86, 1.14, "Valid ports, no duplicate use", fontsize=7.8,
         ha="center", color=MUTED)
    # Eight equally spaced stems are probes, not an invented measured spectrum.
    bx, by = XS[2] + 0.23, 2.34
    ax.plot([bx, bx + 1.26], [by, by], color=NAVY, lw=0.9)
    for i in range(8):
        xx = bx + 0.06 + i * 1.14 / 7
        ax.plot([xx, xx], [by, by + 0.39], color=TEAL, lw=1.6)
        ax.plot(xx, by + 0.39, "o", ms=2.7, color=TEAL)
    text(ax, XS[2] + 0.86, 2.13, "1.51–1.59 μm", ha="center", fontsize=8,
         color=MUTED)
    text(ax, XS[2] + 0.86, 1.70, "8 probes: finite |S|²", ha="center",
         fontsize=9, fontweight="bold")
    text(ax, XS[2] + 0.86, 1.39, "Admission check only", ha="center", fontsize=8.5)
    text(ax, XS[2] + 0.86, 1.14, "Not spectrum equality", ha="center", fontsize=8,
         color=MUTED)
    for i in range(2):
        arrow(ax, XS[i] + CW[i] + 0.02, 2.04, XS[i + 1] - 0.03, 2.04,
              lw=1.1, ms=9)
    # A separate output fork keeps prompt and reference roles unambiguous.
    text(ax, 6.32, 3.43, "Then write", ha="center", fontsize=10.5,
         fontweight="bold")
    text(ax, 6.32, 3.16, "the prose", ha="center", fontsize=10.5,
         fontweight="bold")
    arrow(ax, 5.72, 2.04, 5.90, 2.04, lw=1.1, ms=9)
    rounded(ax, 5.94, 2.12, 0.94, 0.71, WHITE, ec=TEAL, radius=0.07)
    text(ax, 6.41, 2.60, "Prompt", fontsize=9.3, ha="center", color=TEAL,
         fontweight="bold")
    text(ax, 6.41, 2.32, "shown", fontsize=8, ha="center", color=MUTED)
    rounded(ax, 5.94, 1.07, 0.94, 0.71, WHITE, ec=MUTED, radius=0.07,
            ls=(0, (3, 2)))
    text(ax, 6.41, 1.55, "JSON", fontsize=9.3, ha="center", fontweight="bold")
    text(ax, 6.41, 1.28, "stored only", fontsize=8, ha="center", color=MUTED)
    ax.plot([5.83, 5.83, 5.92], [2.04, 2.48, 2.48], color=GRAY, lw=0.8)
    ax.plot([5.83, 5.83, 5.92], [2.04, 1.42, 1.42], color=GRAY, lw=0.8)
    ax.plot([0.15, 6.85], [0.65, 0.65], color=GRAY, lw=0.7)
    text(ax, 0.16, 0.42, "Widths 4 and 8 excluded  ·  No exact expert-graph overlap",
         fontsize=8.5, color=NAVY)
    text(ax, 0.16, 0.17, "Test failures inform the families; zero exact overlap does not rule out near-duplicates.",
         fontsize=8, color=MUTED)
    save(fig, "fig2_failure_verbalization")


if __name__ == "__main__":
    setup()
    build()
