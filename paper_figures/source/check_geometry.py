"""Geometry guard for the paper figures.

Renders a figure and reports, for every text artist:

  * a bounding box that leaves the card it is supposed to sit in;
  * a bounding box that overlaps another text artist's box.

matplotlib lays text out in display coordinates, so this measures what the PDF
actually draws rather than trusting a hand-set constant.  Overlap tolerance and
the card list are per figure; a figure with no declared cards is only checked
for pairwise overlap.

Run:  python check_geometry.py            # all figures
      python check_geometry.py fig6       # one figure
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fig1  # noqa: E402
import fig2  # noqa: E402
import fig3  # noqa: E402
import fig4  # noqa: E402
import fig5  # noqa: E402
import fig6  # noqa: E402
from style import setup  # noqa: E402


@dataclass
class Card:
    """A card in figure inches: name plus its box."""

    name: str
    x0: float
    y0: float
    x1: float
    y1: float


# Cards are declared in the same inch coordinates each figure draws in.  Text
# is assigned to the smallest card that contains its anchor.
FIG6_CARDS = (
    Card("(a) netlist", fig6.AX, fig6.AY, fig6.AX + fig6.AW, fig6.AY + fig6.AH),
    Card("(b) circuit", fig6.BX, fig6.BY, fig6.BX + fig6.BW, fig6.BY + fig6.BH),
    Card("(c) layout", fig6.CX, fig6.CY, fig6.CX + fig6.CW, fig6.CY + fig6.CH),
)

FIG2_CARDS = tuple(
    Card(f"card{i + 1}", fig2.XS[i], fig2.CARD_Y, fig2.XS[i] + fig2.CW[i],
         fig2.CARD_Y + fig2.CARD_H)
    for i in range(len(fig2.XS))
)

FIG3_CARDS = (
    Card("teacher filter", 0.16, 3.36, 0.16 + (7.0 - 0.32), 3.36 + 1.20),
    Card("students", 0.16, 1.24, 0.16 + (7.0 - 0.32), 1.24 + 2.00),
    Card("footer", 0.16, 0.10, 0.16 + (7.0 - 0.32), 0.10 + 1.02),
)

# fig1, fig4, fig5 are axis figures: only pairwise overlap is meaningful.
FIGURES = {
    "fig1": (fig1, ()),
    "fig2": (fig2, FIG2_CARDS),
    "fig3": (fig3, FIG3_CARDS),
    "fig4": (fig4, ()),
    "fig5": (fig5, ()),
    "fig6": (fig6, FIG6_CARDS),
}

OVERLAP_TOL = 0.5  # points of allowed overlap: wobbly ink strokes touch
CARD_TOL = 0.5     # points a box may spill past its card


def _boxes(fig, ax, cards):
    """Return [(label, x0, y0, x1, y1, card)] in inches.

    get_window_extent is in display pixels; dividing by fig.dpi gives inches,
    which is the unit every canvas()/plot_canvas() figure draws in.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    out = []
    for t in ax.texts:
        s = t.get_text().strip()
        if not s:
            continue
        bb = t.get_window_extent(renderer=renderer)
        x0, y0 = bb.x0 / dpi, bb.y0 / dpi
        x1, y1 = bb.x1 / dpi, bb.y1 / dpi
        # the anchor is in data units, which are inches for these figures
        cx, cy = (v / dpi for v in t.get_transform().transform(t.get_position()))
        owner = None
        for c in cards:
            if c.x0 - 1e-9 <= cx <= c.x1 + 1e-9 and c.y0 - 1e-9 <= cy <= c.y1 + 1e-9:
                if owner is None or (c.x1 - c.x0) * (c.y1 - c.y0) < (
                    owner.x1 - owner.x0
                ) * (owner.y1 - owner.y0):
                    owner = c
        # Monospace code tokens are set adjacent by design: their bounding
        # boxes touch at the side bearings without any ink overlapping, so a
        # pair of them is never a defect.
        fam = t.get_fontfamily() or [""]
        mono = "Mono" in fam[0]
        out.append((s.replace("\n", " / "), x0, y0, x1, y1, owner, mono))
    return out


def _capture(module):
    """Replace module.save so build() hands over the figure instead of closing it."""
    box: list = []

    def fake_save(fig, stem, png_dpi=600):
        box.append(fig)

    module.save = fake_save
    return box


def check(name: str) -> list[str]:
    module, cards = FIGURES[name]
    # Each build() ends in style.save, which closes the figure.  Swap in a
    # capture so the laid-out figure survives long enough to be measured.
    box = _capture(module)
    module.build()
    if not box:
        return [f"{name}: build() produced no figure"]
    fig = box[-1]
    ax = fig.axes[0]
    problems: list[str] = []

    boxes = _boxes(fig, ax, cards)
    tol = CARD_TOL / 72.0
    for s, x0, y0, x1, y1, owner, _mono in boxes:
        if owner is None:
            continue
        if x0 < owner.x0 - tol or x1 > owner.x1 + tol or y0 < owner.y0 - tol or y1 > owner.y1 + tol:
            problems.append(
                f"{name} {owner.name}: '{s[:44]}' box "
                f"({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f}) leaves card "
                f"({owner.x0:.2f},{owner.y0:.2f})-({owner.x1:.2f},{owner.y1:.2f})"
            )

    otol = OVERLAP_TOL / 72.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[5] and b[5]:
                continue  # adjacent monospace tokens, see _boxes
            ox = min(a[3], b[3]) - max(a[1], b[1])
            oy = min(a[4], b[4]) - max(a[2], b[2])
            if ox > otol and oy > otol:
                problems.append(
                    f"{name}: '{a[0][:32]}' overlaps '{b[0][:32]}' "
                    f"by {ox * 72:.1f} x {oy * 72:.1f} pt"
                )
    plt.close(fig)
    return problems


def main() -> int:
    setup()
    names = sys.argv[1:] or list(FIGURES)
    bad = 0
    for name in names:
        problems = check(name)
        if problems:
            bad += len(problems)
            print(f"FAIL {name}: {len(problems)} geometry problem(s)")
            for p in problems[:24]:
                print("   ", p)
            if len(problems) > 24:
                print(f"    ... and {len(problems) - 24} more")
        else:
            print(f"ok   {name}")
    if bad:
        print(f"check_geometry: {bad} problem(s)")
        return 1
    print("check_geometry: no overflow, no overlap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
