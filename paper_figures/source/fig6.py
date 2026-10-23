"""Figure 6. The scored object is the JSON netlist, not the mask.

Three hand-drawn panels in the same visual language as fig1..fig5:

  (a) the netlist -- the JSON PICBench scores.  instances, connections and
      ports are drawn in full; the models map is elided.
  (b) the circuit -- the graph that JSON names, with the phase setting and the
      arm lengths on the elements that carry them.
  (c) the chip layout -- a schematic mask of the same topology, which no
      spectrum match places, routes, or checks against a PDK cell.

The netlist is an illustration of one Mach-Zehnder, written in PICBench's own
schema (compare testcases/MZM/MZM_ref.json): a top-level "netlist" holding
"instances", "connections" keyed "left,port": "right,port", and "ports" binding
the circuit's external pins; an instance is either a bare model name or
{"component", "settings"}; a sibling "models" map resolves those names to SAX
functions.  Here an mmi1x2 splitter and combiner, one straight_heat_metal phase
shifter at 90 on the top arm, and straight arms of 55.0 and 45.0 um (a 10 um
path difference).  It is not one of the 24 expert files and it was not passed
through SAX.

Type budget: the listing is 6.6 pt monospace on ~8.0 pt leading, the smallest
drawn type in the figure set is 6.0 pt, and every glyph is vector, so the
panel stays sharp at any zoom.  The previous raster was 241 dpi with ~3.3 pt
listing type.
"""

from __future__ import annotations

import re

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MPath
from matplotlib.patches import Circle, PathPatch, Polygon
from matplotlib.textpath import TextPath

from style import (
    AMBER,
    BLUE,
    BLUE_SOFT,
    GRAY,
    GRAY_SOFT,
    MUTED,
    NAVY,
    TEAL,
    WHITE,
    arrow,
    canvas,
    rect,
    rounded,
    save,
    setup,
    text,
)

W, H = 7.0, 4.6

# Card geometry.  (a) is tall and narrow because a JSON listing needs lines,
# not width; (b) and (c) stack to its right because both read left to right.
# AW is set by the listing: its longest line is 55 monospace characters, which
# is 3.035 in at CODE_SIZE, plus the 0.15 in left inset and 0.10 in right margin.
AX, AY, AW, AH = 0.12, 0.20, 3.32, 4.20
BX, BY, BW, BH = 3.68, 2.42, 3.20, 1.98
CX, CY, CW, CH = 3.68, 0.20, 3.20, 1.98

# Small type needs more ink than the pastel fills do, so the listing colors
# are darker than the palette entries they belong to.
CODE_KEY = "#2E5F8A"
CODE_STR = "#1E7A6B"
CODE_NUM = "#BC5A57"
CODE_PUN = MUTED
CODE_SIZE = 6.6
CODE_FONT = "DejaVu Sans Mono"
# Small type on white needs more ink than a swatch does.
CORAL_D = "#B45653"
# Nothing in this figure set is drawn below this size.  At width=\linewidth the
# 7.0 in canvas is scaled to 6.94 in, so MIN_TYPE lands at 5.95 pt on the page.
MIN_TYPE = 6.0

# One Mach-Zehnder in PICBench's schema.  Indentation is two spaces per level;
# the longest line (45 characters) sets the listing width, and build() raises if
# a line ever overruns the card.
LISTING = (
    '{',
    '  "netlist": {',
    '    "instances": {',
    '      "splitter": "mmi",',
    '      "combiner": "mmi",',
    '      "ps_top": {',
    '        "component": "straight_heat_metal",',
    '        "settings": {"length": 55.0, "phase_shift": 90}',
    '      },',
    '      "ps_bot": {',
    '        "component": "straight",',
    '        "settings": {"length": 45.0}',
    '      }',
    '    },',
    '    "connections": {',
    '      "splitter,O1": "ps_top,I1",',
    '      "splitter,O2": "ps_bot,I1",',
    '      "ps_top,O1": "combiner,O1",',
    '      "ps_bot,O1": "combiner,O2"',
    '    },',
    '    "ports": {',
    '      "I1": "splitter,I1",',
    '      "O1": "combiner,I1"',
    '    }',
    '  },',
    '  "models": {...}',
    '}',
)

_TOKEN = re.compile(
    r'(?P<key>"(?:[^"\\]|\\.)*"(?=\s*:))'
    r'|(?P<str>"(?:[^"\\]|\\.)*")'
    r'|(?P<num>-?\d+(?:\.\d+)?)'
    r'|(?P<ell>\{\.\.\.\})'
    r'|(?P<pun>\s+|.)'
)


def _mono_advance(size: float) -> float:
    """Advance of one monospace character, in inches."""
    prop = FontProperties(family=CODE_FONT, size=size)
    one = TextPath((0, 0), "M", prop=prop).get_extents().width
    two = TextPath((0, 0), "MM", prop=prop).get_extents().width
    return (two - one) / 72.0


def _tokens(line: str):
    for m in _TOKEN.finditer(line):
        kind = m.lastgroup
        color = {
            "key": CODE_KEY,
            "str": CODE_STR,
            "num": CODE_NUM,
            "ell": MUTED,
            "pun": CODE_PUN,
        }[kind]
        yield m.group(), color


def badge(ax, x, y, letter, accent) -> None:
    ax.add_patch(Circle((x, y), 0.145, facecolor=accent, edgecolor=NAVY, lw=0.8, zorder=4))
    text(ax, x, y - 0.012, letter, fontsize=8.2, color=WHITE, fontweight="bold",
         ha="center", va="center", zorder=5)


def panel_a(ax) -> None:
    rounded(ax, AX, AY, AW, AH, WHITE, ec=NAVY, lw=1.0, radius=0.08)
    badge(ax, AX + 0.26, AY + AH - 0.26, "a", AMBER)
    text(ax, AX + 0.48, AY + AH - 0.175, "the netlist", fontsize=8.6, fontweight="bold", va="center")
    text(ax, AX + 0.48, AY + AH - 0.345, "(JSON)", fontsize=8.6, fontweight="bold", va="center")

    # The listing fills the card below the title.  Leading is derived from the
    # available height so a line is never added without shrinking the type.
    top, bottom = AY + AH - 0.50, AY + 0.16
    lead = (top - bottom) / float(len(LISTING))
    adv = _mono_advance(CODE_SIZE)
    x0 = AX + 0.15
    widest = 0.0
    for i, line in enumerate(LISTING):
        y = top - (i + 0.5) * lead
        x = x0
        for chunk, color in _tokens(line):
            if chunk.strip():
                ax.text(x, y, chunk, fontsize=CODE_SIZE, family=CODE_FONT,
                        color=color, ha="left", va="center", zorder=5)
            x += adv * len(chunk)
        widest = max(widest, x)
    if widest > AX + AW - 0.10:
        raise SystemExit(f"fig6: listing overruns the card by {widest - (AX + AW - 0.10):.3f} in")


# --- panel (b): the circuit graph -------------------------------------------

def guide(ax, pts, lw=1.5, color=NAVY) -> None:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, solid_joinstyle="round",
            solid_capstyle="round", zorder=4)


def mmi(ax, x, y, w, h, split=True) -> None:
    """Tapered MMI: narrow on the single-port side, wide on the two-port side."""
    lo, hi = y + 0.15 * h, y + 0.85 * h
    if split:  # narrow left, wide right
        pts = [(x, lo + 0.06 * h), (x + w, y), (x + w, y + h), (x, hi - 0.06 * h)]
    else:      # wide left, narrow right
        pts = [(x, y), (x + w, lo + 0.06 * h), (x + w, hi - 0.06 * h), (x, y + h)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=BLUE, edgecolor=NAVY,
                         lw=0.7, zorder=5))


def port(ax, x, y, name, va="center") -> None:
    ax.add_patch(Circle((x, y), 0.035, facecolor=TEAL, edgecolor=NAVY, lw=0.6, zorder=6))
    text(ax, x - 0.07, y, name, fontsize=6.4, fontweight="bold", ha="right",
         va=va, color=NAVY)


def elem_label(ax, x, y, name, param, above) -> None:
    d = 0.10 if above else -0.10
    e = 0.195 if above else -0.195
    text(ax, x, y + d, name, fontsize=6.5, fontweight="bold", ha="center", va="center")
    text(ax, x, y + e, param, fontsize=MIN_TYPE, color=MUTED, ha="center", va="center")


def panel_b(ax) -> None:
    rounded(ax, BX, BY, BW, BH, BLUE_SOFT, ec=NAVY, lw=1.0, radius=0.08)
    badge(ax, BX + 0.26, BY + BH - 0.26, "b", BLUE)
    text(ax, BX + 0.48, BY + BH - 0.175, "the circuit", fontsize=8.6, fontweight="bold", va="center")
    text(ax, BX + 0.48, BY + BH - 0.345, "(graph)", fontsize=8.6, fontweight="bold", va="center")

    x0, y0 = BX + 0.12, BY + 0.12
    y_t, y_b = y0 + 0.96, y0 + 0.24
    y_hi, y_lo = y0 + 0.78, y0 + 0.42           # mmi port heights
    cy_arm = y0 + 0.60                          # external port height
    x_in = x0 + 0.26
    x_m1, x_m1e = x0 + 0.44, x0 + 0.78
    x_a0, x_a1 = x0 + 0.94, x0 + 2.02           # one element per arm
    x_m2, x_m2e = x0 + 2.18, x0 + 2.52
    x_out = x0 + 2.70

    # ports: the netlist binds one pin in and one pin out.
    port(ax, x_in, cy_arm, "I1")
    guide(ax, [(x_in, cy_arm), (x_m1, cy_arm)])
    mmi(ax, x_m1, y0 + 0.36, 0.34, 0.48, split=True)
    text(ax, (x_m1 + x_m1e) / 2, y0 + 0.20, "splitter", fontsize=6.5, fontweight="bold",
         ha="center", va="center")

    # splitter,O1 / O2 to the two arms, then the arms to the combiner.
    guide(ax, [(x_m1e, y_hi), (x_a0, y_t)])
    guide(ax, [(x_m1e, y_lo), (x_a0, y_b)])
    guide(ax, [(x_a1, y_t), (x_m2, y_hi)])
    guide(ax, [(x_a1, y_b), (x_m2, y_lo)])

    mmi(ax, x_m2, y0 + 0.36, 0.34, 0.48, split=False)
    text(ax, (x_m2 + x_m2e) / 2, y0 + 0.20, "combiner", fontsize=6.5, fontweight="bold",
         ha="center", va="center")
    guide(ax, [(x_m2e, cy_arm), (x_out, cy_arm)])
    ax.add_patch(Circle((x_out, cy_arm), 0.035, facecolor=TEAL, edgecolor=NAVY, lw=0.6, zorder=6))
    text(ax, x_out + 0.07, cy_arm, "O1", fontsize=6.4, fontweight="bold", va="center")

    # the top arm carries the heater; the bottom arm is plain straight.
    x_c = (x_a0 + x_a1) / 2
    rect(ax, x_a0, y_t - 0.055, x_a1 - x_a0, 0.11, AMBER, ec=NAVY, lw=0.5, z=6)
    guide(ax, [(x_a0, y_t), (x_a1, y_t)], lw=1.2)
    rect(ax, x_a0, y_b - 0.055, x_a1 - x_a0, 0.11, WHITE, ec=NAVY, lw=0.5, z=6)
    guide(ax, [(x_a0, y_b), (x_a1, y_b)], lw=1.2)

    elem_label(ax, x_c, y_t, "ps_top", "heat_metal, 55.0, phase 90", above=True)
    elem_label(ax, x_c, y_b, "ps_bot", "straight, 45.0", above=False)


# --- panel (c): the chip layout --------------------------------------------

def grating(ax, cx, cy) -> None:
    w, h = 0.20, 0.30
    rounded(ax, cx - w / 2, cy - h / 2, w, h, BLUE_SOFT, ec=NAVY, lw=0.7, radius=0.02, z=5)
    for i in range(5):
        xx = cx - w / 2 + 0.028 + i * 0.036
        ax.plot([xx, xx], [cy - h / 2 + 0.04, cy + h / 2 - 0.04],
                color=NAVY, lw=0.5, zorder=6)


def sbend(ax, x0, y0, x1, y1, lw=1.8) -> None:
    xm = (x0 + x1) / 2.0
    path = MPath(
        [(x0, y0), (xm, y0), (xm, y1), (x1, y1)],
        [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4],
    )
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=NAVY, lw=lw,
                           capstyle="round", zorder=4))


def panel_c(ax) -> None:
    rounded(ax, CX, CY, CW, CH, WHITE, ec=NAVY, lw=1.0, radius=0.08)
    badge(ax, CX + 0.26, CY + CH - 0.26, "c", GRAY)
    text(ax, CX + 0.48, CY + CH - 0.175, "the chip layout", fontsize=8.6, fontweight="bold", va="center")
    text(ax, CX + 0.48, CY + CH - 0.345, "(not scored)", fontsize=8.6, fontweight="bold",
         va="center", color=CORAL_D)

    # substrate: leaves a caption band below it inside the card
    sx0, sy0, sx1, sy1 = CX + 0.12, CY + 0.26, CX + CW - 0.12, CY + CH - 0.48
    rounded(ax, sx0, sy0, sx1 - sx0, sy1 - sy0, GRAY_SOFT, ec=GRAY, lw=0.8, radius=0.05, z=2)

    cy = (sy0 + sy1) / 2.0
    y_t, y_b = sy1 - 0.34, sy0 + 0.34
    x_in, x_out = sx0 + 0.26, sx1 - 0.26
    x_b0, x_b1 = x_in + 0.46, x_out - 0.46
    x_mid = (x_b0 + x_b1) / 2.0

    grating(ax, x_in, cy)
    grating(ax, x_out, cy)
    arrow(ax, sx0 + 0.02, cy, x_in - 0.12, cy, color=MUTED, lw=0.9, ms=6, z=5)

    sbend(ax, x_in + 0.10, cy, x_b0, y_t)
    sbend(ax, x_in + 0.10, cy, x_b0, y_b)
    ax.plot([x_b0, x_b1], [y_t, y_t], color=NAVY, lw=1.8, solid_capstyle="round", zorder=4)
    ax.plot([x_b0, x_b1], [y_b, y_b], color=NAVY, lw=1.8, solid_capstyle="round", zorder=4)
    sbend(ax, x_b1, y_t, x_out - 0.10, cy)
    sbend(ax, x_b1, y_b, x_out - 0.10, cy)

    # heater over the top arm, fed by a metal pad
    hx0, hx1 = x_mid - 0.38, x_mid + 0.38
    rect(ax, hx0, y_t - 0.055, hx1 - hx0, 0.11, AMBER, ec=NAVY, lw=0.5, z=6)
    px0, py0, pw, ph = hx0 + 0.18, y_t + 0.10, 0.40, 0.12
    rounded(ax, px0, py0, pw, ph, BLUE, ec=NAVY, lw=0.6, radius=0.02, z=6)
    ax.plot([px0 + pw / 2, px0 + pw / 2], [py0, y_t + 0.055], color=NAVY, lw=0.8, zorder=5)
    text(ax, hx1 + 0.07, py0 + ph / 2, "metal pad", fontsize=6.0, color=NAVY,
         ha="left", va="center")

    # The band between the two arms is empty, so both device names go there:
    # the heater named under the arm it sits on, the core named over its line.
    text(ax, x_mid, cy + 0.10, "heater (phase shifter)", fontsize=6.0,
         color=NAVY, ha="center", va="center")
    text(ax, x_mid, cy - 0.12, "the navy line is the waveguide core", fontsize=6.0,
         color=NAVY, ha="center", va="center")

    for x, s in ((x_in, "input coupler"), (x_mid, "silicon substrate"),
                 (x_out, "output coupler")):
        weight = "bold" if s == "silicon substrate" else "normal"
        text(ax, x, CY + 0.13, s, fontsize=6.0, color=MUTED if weight == "normal" else NAVY,
             ha="center", va="center", fontweight=weight)


def flow(ax) -> None:
    mid = BX + BW / 2.0
    arrow(ax, AX + AW + 0.02, 3.41, BX - 0.02, 3.41, color=NAVY, lw=1.2, ms=9)
    arrow(ax, mid, BY - 0.02, mid, CY + CH + 0.02, color=NAVY, lw=1.2, ms=9)


def build() -> None:
    fig, ax = canvas(W, H)
    panel_a(ax)
    panel_b(ax)
    panel_c(ax)
    flow(ax)
    save(fig, "fig6_netlist_layout")


if __name__ == "__main__":
    setup()
    build()
