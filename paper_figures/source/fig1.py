"""Figure 1. A simulable netlist is not a spectrum match.

Hand-drawn (FEAST-style) stacked bars: a 0--100% bar per model, one mutually
exclusive stage per sample, with a labelled "SAX starts" boundary and an
enlarged inset for Qwen's four simulable samples.

Counts are from paper Table loci (120 samples per model).
  Qwen3.5-9B:       no result 7 + json.loads 72 + SAX/schema 37 + mismatch 3 + match 1
  DeepSeek-V4-Pro:  1 + 0 + 44 + 37 + 38
The first two Qwen rows collapse to "JSON not loaded" (79); DeepSeek's is 1.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from style import (
    AMBER,
    BLUE,
    CORAL,
    GRAY,
    MUTED,
    NAVY,
    TEAL,
    WHITE,
    curve_arrow,
    save,
    setup,
    text,
)

W, H = 7.0, 4.35

STAGES = (
    ("JSON not loaded", (79, 1), AMBER, "..", NAVY),
    ("SAX / schema reject", (37, 44), BLUE, "///", WHITE),
    ("Spectrum mismatch", (3, 37), CORAL, "xx", NAVY),
    ("Spectrum match", (1, 38), TEAL, None, WHITE),
)

MODELS = ("Qwen3.5-9B", "DeepSeek-V4-Pro")
# simulable begins after "JSON not loaded" + "SAX / schema reject"
SYNTAX_PCT = (96.67, 37.5)


def build() -> None:
    fig, ax = plt.subplots(figsize=(W, H), dpi=120)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    fig.subplots_adjust(left=0.245, bottom=0.30, right=0.97, top=0.94)

    ypos = [1, 0]
    left = [0.0, 0.0]
    for name, counts, color, hatch, ink in STAGES:
        pcts = [c / 1.2 for c in counts]  # 120 samples -> 100%
        ax.barh(
            ypos,
            pcts,
            left=left,
            height=0.52,
            color=color,
            edgecolor=NAVY,
            lw=0.7,
            hatch=hatch,
            zorder=3,
        )
        for i, y in enumerate(ypos):
            c = counts[i]
            if c >= 8:
                ax.text(
                    left[i] + pcts[i] / 2.0,
                    y,
                    str(c),
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=ink,
                    fontweight="bold",
                    zorder=4,
                )
        left = [l + p for l, p in zip(left, pcts)]

    # Navy "SAX starts" boundary per model: where the simulable region begins.
    for y, xb in zip(ypos, SYNTAX_PCT):
        ax.plot([xb, xb], [y + 0.27, y + 0.55], color=NAVY, lw=1.4, zorder=5)
        ax.text(
            xb, y + 0.62, "SAX starts", ha="center", va="bottom",
            fontsize=7.8, color=NAVY, fontweight="bold", zorder=5,
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(MODELS, fontsize=10, color=NAVY, fontweight="bold")
    ax.set_ylim(-0.72, 1.95)
    ax.tick_params(axis="y", which="both", left=False, length=0)

    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=8.5)
    ax.tick_params(axis="x", length=4, width=0.8)
    ax.set_xlabel("Percent of 120 Without-EF samples", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NAVY)
        ax.spines[side].set_linewidth(0.9)
    ax.grid(True, axis="x", color=GRAY, lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Conditional match rate, above the Qwen bar (empty band).
    text(
        ax, 1.0, 1.72,
        "Conditional on a spectrum:  Qwen 1/4  ·  DeepSeek 38/75",
        fontsize=7.8, color=MUTED, ha="left", va="center",
    )

    # Legend (four stages), hand-drawn swatches.
    handles = []
    for name, _, color, hatch, ink in STAGES:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=NAVY, lw=0.6, hatch=hatch, label=name)
        )
    leg = ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.44, -0.28),
        ncol=2,
        fontsize=8.4,
        frameon=True,
        framealpha=1.0,
        edgecolor=NAVY,
        handlelength=1.35,
        handleheight=1.0,
        columnspacing=1.6,
        handletextpad=0.55,
        borderpad=0.6,
    )
    leg.get_frame().set_linewidth(0.8)

    # Enlarged inset for Qwen's four simulable samples, off the 0--100 axis.
    text(ax, 102.0, 1.62, "Qwen's 4 simulable\nsamples, enlarged", ha="left", va="top", fontsize=7.4, color=MUTED)
    cells = ((CORAL, "3"), (CORAL, ""), (CORAL, ""), (TEAL, "1"))
    ix, cell = 102.0, 3.7
    iy = 0.78
    for k, (c, lab) in enumerate(cells):
        ax.add_patch(
            Rectangle(
                (ix + k * cell, iy), cell - 0.5, 0.34,
                facecolor=c, edgecolor=NAVY, lw=0.7, zorder=4,
            )
        )
        if lab:
            ax.text(
                ix + k * cell + (cell - 0.5) / 2, iy + 0.17, lab,
                ha="center", va="center", fontsize=7.4, color=NAVY, fontweight="bold", zorder=5,
            )
    curve_arrow(ax, 98.6, 0.98, 101.4, 0.96, bend=0.10, color=MUTED, lw=0.9, ms=7)
    text(ax, 102.0, 0.52, "mismatch (3) + match (1)", ha="left", va="center", fontsize=7.2, color=MUTED)
    text(ax, 102.0, 0.28, "not part of the 0–100 axis", ha="left", va="center", fontsize=7.2, color=MUTED)

    save(fig, "fig1_failure_audit")


if __name__ == "__main__":
    setup()
    build()
