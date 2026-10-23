"""Shared hand-drawn style for the PICBench paper figures.

Visual language follows the FEAST paper (10487_FEAST_Aligning_Agent_Fai.pdf):
rounded ink-outlined boxes, soft pastel fills, hand-written labels.
Implemented with matplotlib's ``path.sketch`` wobble (deterministic: the
sketch RNG is seeded from the path, so rebuilds are byte-identical across
processes) and the Comic Neue hand-drawn typeface, with DejaVu Sans as a
per-character fallback for glyphs Comic Neue lacks (mu, >=, |, superscripts).

Coordinates in each figure are inches on a white canvas.
Fonts live in ``paper/fonts/`` so the figures are reproducible offline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "paper" / "fonts"
OUT = Path("/tmp/paper_figures")

NAVY = "#24354B"
BLUE = "#4B7FA4"
BLUE_SOFT = "#D5E3EE"
TEAL = "#2A9988"
TEAL_SOFT = "#D5EFEA"
CORAL = "#D8807E"
CORAL_SOFT = "#F6E4E3"
AMBER = "#E5A85C"
AMBER_DEEP = "#C4893E"
AMBER_SOFT = "#FAEBCF"
GRAY = "#C6CED8"
GRAY_SOFT = "#EEF1F4"
WHITE = "#FFFFFF"
MUTED = "#5C6B7A"
INK = NAVY

# Locus colors stay inside the same semantic families as the audit.
LOCUS_COLOR = {
    "Ellipsis": AMBER,
    "Comment": AMBER_DEEP,
    "Wrong model": BLUE,
    "Ports": "#7FA6C4",
    "Functional": CORAL,
    "Graft": "#B56866",
}

# Hand-drawn wobble: scale, length, randomness.  Tuned for 7-inch figures
# dense with bars and small type: visible, but it does not blur 3/120 slivers.
SKETCH = (0.62, 110.0, 1.45)

_families_added = False


def _add_fonts() -> None:
    global _families_added
    if _families_added:
        return
    for path in sorted(FONT_DIR.glob("*.ttf")):
        fm.fontManager.addfont(str(path))
    _families_added = True


def setup() -> None:
    """Register fonts and the global hand-drawn rcParams. Call once per run."""
    _add_fonts()
    mpl.rcParams.update(
        {
            # Per-character fallback: DejaVu Sans supplies mu, >=, |, ², ...
            "font.family": ["Comic Neue", "DejaVu Sans"],
            "font.size": 8,
            "text.color": NAVY,
            "path.sketch": SKETCH,
            "axes.linewidth": 0.9,
            "axes.edgecolor": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "lines.linewidth": 1.0,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "hatch.linewidth": 0.55,
            "hatch.color": NAVY,
            "axes.unicode_minus": False,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def canvas(w: float, h: float, sketch_axes: bool = False):
    """Inch-coordinate canvas filling the whole figure.

    ``sketch_axes`` keeps the drawn spines off; figures that need axis frames
    use ``plot_canvas`` instead.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(w, h), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    fig.patch.set_facecolor(WHITE)
    return fig, ax


def plot_canvas(w: float, h: float):
    """Real axes canvas for scatter/bar figures that keep tick labels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(w, h), dpi=120)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def text(ax, x, y, s, **kwargs):
    kwargs.setdefault("color", NAVY)
    kwargs.setdefault("ha", "left")
    kwargs.setdefault("va", "center")
    kwargs.setdefault("fontsize", 8)
    kwargs.setdefault("linespacing", 1.25)
    kwargs.setdefault("zorder", 5)
    return ax.text(x, y, s, **kwargs)


def rounded(ax, x, y, w, h, fc, ec=NAVY, lw=0.9, radius=0.06, ls="-", z=2, hatch=None):
    """Hand-drawn rounded box (FEAST-style card)."""
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        mutation_aspect=1,
        hatch=hatch,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def rect(ax, x, y, w, h, fc, ec=WHITE, lw=0.5, z=3, hatch=None):
    from matplotlib.patches import Rectangle

    patch = Rectangle(
        (x, y),
        w,
        h,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        hatch=hatch,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.0, style="-|>", z=4, ms=8):
    """Wobbly hand-drawn arrow."""
    from matplotlib.patches import FancyArrowPatch

    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=ms,
        color=color,
        linewidth=lw,
        shrinkA=0,
        shrinkB=0,
        joinstyle="round",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def curve_arrow(ax, x1, y1, x2, y2, *, bend=0.25, color=NAVY, lw=1.0, ms=8, z=4):
    """Curved hand-drawn arrow (FEAST uses these between cards)."""
    from matplotlib.patches import FancyArrowPatch

    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        connectionstyle=f"arc3,rad={bend}",
        arrowstyle="-|>",
        mutation_scale=ms,
        color=color,
        linewidth=lw,
        shrinkA=2,
        shrinkB=2,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def save(fig, stem: str, png_dpi: int = 600) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", format="svg")
    fig.savefig(OUT / f"{stem}.pdf", format="pdf")
    fig.savefig(OUT / f"{stem}.png", format="png", dpi=png_dpi)
    fig.savefig(OUT / f"{stem}.preview.png", format="png", dpi=140)
    import matplotlib.pyplot as plt

    plt.close(fig)
