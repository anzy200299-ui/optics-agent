"""Five-stage overview of the paper.

Solid panels are measurements. The last panel is dashed because neither
student has a PICBench score. Counts are the constants checked in
draw_paper_figures.py against the audit logs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draw_paper_figures as base

NAVY = base.NAVY
BLUE = base.BLUE
TEAL = base.TEAL
ORANGE = base.ORANGE
CORAL = base.CORAL
WHITE = base.WHITE
mix = base.mix

INK = NAVY
MUTED = mix(NAVY, WHITE, 0.42)
RULE = mix(NAVY, WHITE, 0.72)

PANELS = [
    ("A", "Audit", "Two-stage judge", mix(BLUE, WHITE, 0.78), BLUE),
    ("B", "Verbalize", "Failure loci", mix(TEAL, WHITE, 0.80), TEAL),
    ("C", "Questions", "1,000 SAX checks", mix(ORANGE, WHITE, 0.72), mix(ORANGE, NAVY, 0.25)),
    ("D", "Teacher", "985 kept", mix(CORAL, WHITE, 0.78), CORAL),
    ("E", "Test", "Not scored", mix(NAVY, WHITE, 0.90), mix(NAVY, WHITE, 0.45)),
]


def rbox(ax, x, y, w, h, *, fc, ec, lw=0.7, r=0.05, ls="-"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
            joinstyle="round",
            zorder=1,
        )
    )


def card(ax, x, y, w, h, *, fc=WHITE, ec=RULE, lw=0.6):
    rbox(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=0.04)


def label(ax, x, y, s, *, size=6.2, weight="normal", color=INK, ha="left", va="center", style="normal"):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        fontstyle=style,
        color=color,
        ha=ha,
        va=va,
        linespacing=1.15,
        zorder=4,
    )


def chevron(ax, x, y, color):
    ax.add_patch(
        Polygon(
            [(x, y - 0.055), (x + 0.055, y), (x, y + 0.055)],
            closed=True,
            facecolor=color,
            edgecolor="none",
            zorder=5,
        )
    )


def stack(ax, x, y, w, h, parts):
    total = sum(value for value, _ in parts)
    cursor = x
    for value, color in parts:
        seg = w * value / total
        if seg <= 0:
            continue
        ax.add_patch(
            Rectangle(
                (cursor, y),
                seg,
                h,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.3,
                zorder=3,
            )
        )
        cursor += seg


def draw() -> None:
    fig_w, fig_h = 6.50, 4.88
    fig, ax = base.canvas(fig_w, fig_h)
    margin = 0.045
    gap = 0.062
    cw = (fig_w - 2 * margin - 4 * gap) / 5
    xs = [margin + i * (cw + gap) for i in range(5)]
    bottom = 0.40
    top = 4.78
    ch = top - bottom

    for i, (x, (letter, title, subtitle, fill, accent)) in enumerate(zip(xs, PANELS)):
        dashed = letter == "E"
        rbox(
            ax,
            x,
            bottom,
            cw,
            ch,
            fc=fill,
            ec=accent,
            lw=0.9 if not dashed else 1.05,
            r=0.07,
            ls=(0, (2.2, 1.4)) if dashed else "-",
        )
        label(ax, x + 0.08, top - 0.16, letter, size=8.2, weight="bold", color=accent)
        label(ax, x + 0.26, top - 0.15, title, size=7.6, weight="bold")
        label(ax, x + 0.08, top - 0.36, subtitle, size=5.8, color=MUTED)
        if i < 4:
            chevron(ax, x + cw + 0.004, top - 0.22, accent)

    inner_w = cw - 0.12

    def ix(i):
        return xs[i] + 0.06

    a = ix(0)
    card(ax, a, 3.42, inner_w, 0.88)
    label(ax, a + 0.06, 4.12, "Qwen3.5-9B", size=6.5, weight="bold")
    label(ax, a + 0.06, 3.92, "7 no span, 72 json.loads", size=5.7)
    label(ax, a + 0.06, 3.74, "4/120 simulable", size=5.7)
    label(ax, a + 0.06, 3.56, "1/120 match", size=5.7, color=base.TEAL_INK, weight="bold")

    card(ax, a, 2.46, inner_w, 0.88)
    label(ax, a + 0.06, 3.16, "DeepSeek-V4-Pro", size=6.4, weight="bold")
    label(ax, a + 0.06, 2.96, "75/120 simulable", size=5.7)
    label(ax, a + 0.06, 2.78, "38/120 match", size=5.7, color=base.TEAL_INK, weight="bold")
    label(ax, a + 0.06, 2.60, "38/75 given a spectrum", size=5.6)

    card(ax, a, 1.62, inner_w, 0.76)
    qwen_parts = [
        (base.QWEN["extract"] + base.QWEN["json"], ORANGE),
        (base.QWEN["sax"], BLUE),
        (base.QWEN["wrong"], CORAL),
        (base.QWEN["match"], TEAL),
    ]
    deep_parts = [
        (base.DEEPSEEK["extract"] + base.DEEPSEEK["json"], ORANGE),
        (base.DEEPSEEK["sax"], BLUE),
        (base.DEEPSEEK["wrong"], CORAL),
        (base.DEEPSEEK["match"], TEAL),
    ]
    label(ax, a + 0.06, 2.22, "Qwen", size=5.4, color=MUTED)
    label(ax, a + 0.06, 1.86, "DeepSeek", size=5.4, color=MUTED)
    stack(ax, a + 0.46, 2.14, inner_w - 0.54, 0.13, qwen_parts)
    stack(ax, a + 0.46, 1.78, inner_w - 0.54, 0.13, deep_parts)

    card(ax, a, 0.52, inner_w, 1.02, fc=mix(CORAL, WHITE, 0.55), ec=mix(CORAL, NAVY, 0.15))
    label(ax, a + inner_w / 2, 1.28, "Syntax ≠ function", size=6.5, weight="bold", ha="center")
    label(ax, a + 0.06, 1.04, "Not loaded, then a wrong", size=5.5)
    label(ax, a + 0.06, 0.86, "spectrum, then a match.", size=5.5)
    label(ax, a + 0.06, 0.66, "Qwen conditional: 1/4.", size=5.5)

    b = ix(1)
    families = [
        ("Ellipsis", 280, ORANGE),
        ("Comment", 160, mix(ORANGE, NAVY, 0.18)),
        ("Wrong model", 280, BLUE),
        ("Ports", 80, mix(BLUE, NAVY, 0.28)),
        ("Functional", 120, CORAL),
        ("Graft", 80, mix(CORAL, NAVY, 0.22)),
    ]
    y = 4.18
    for name, count, color in families:
        ax.add_patch(Rectangle((b, y - 0.07), 0.08, 0.14, facecolor=color, edgecolor="none", zorder=3))
        label(ax, b + 0.14, y, name, size=6.0)
        label(ax, b + inner_w - 0.02, y, str(count), size=6.2, weight="bold", ha="right")
        y -= 0.28
    label(ax, b, 2.42, "9B text-heuristic labels", size=5.6, color=MUTED)
    label(ax, b, 2.24, "Weights were not refit", size=5.6, color=MUTED)
    moves = [
        ("1", "Index by the locus"),
        ("2", "Legal graph, then prose"),
        ("3", "One id: mzi versus mzm"),
    ]
    y = 1.92
    for n, line in moves:
        ax.add_patch(Circle((b + 0.08, y), 0.075, facecolor=TEAL, edgecolor="none", zorder=3))
        label(ax, b + 0.08, y, n, size=5.6, weight="bold", color=WHITE, ha="center")
        label(ax, b + 0.22, y, line, size=5.6)
        y -= 0.36
    label(ax, b, 0.62, "Same wiring, one SAX id", size=5.5, color=MUTED)

    c = ix(2)
    card(ax, c, 3.28, inner_w, 1.02)
    label(ax, c + 0.06, 4.08, "Keep a draw when", size=6.3, weight="bold")
    label(ax, c + 0.06, 3.86, "8 wavelengths are finite", size=5.6)
    label(ax, c + 0.06, 3.68, "in [1.51, 1.59] μm", size=5.6)
    label(ax, c + 0.06, 3.48, "Not the 1,000-point test", size=5.5, color=base.AMBER_INK, weight="bold")

    card(ax, c, 2.18, inner_w, 1.02)
    label(ax, c + 0.06, 2.98, "Prompt, JSON out", size=6.2, weight="bold")
    label(ax, c + 0.06, 2.76, "Prose is the prompt", size=5.6)
    label(ax, c + 0.06, 2.58, "JSON is held out", size=5.6)
    label(ax, c + 0.06, 2.38, "Seed 20261020", size=5.6, color=MUTED)

    card(ax, c, 0.52, inner_w, 1.58)
    label(ax, c + 0.06, 1.90, "Held out of the pool", size=6.2, weight="bold")
    label(ax, c + 0.06, 1.68, "Widths 2, 3, 5, 6 only", size=5.6)
    label(ax, c + 0.06, 1.50, "59, 76, 70, 75 of 280", size=5.6)
    label(ax, c + 0.06, 1.30, "Widths 4 and 8: none", size=5.6)
    label(ax, c + 0.06, 1.10, "24 problems held out", size=5.6)
    label(ax, c + 0.06, 0.90, "8-QAM, 64-QAM, test WDM", size=5.5)
    label(ax, c + 0.06, 0.68, "Fingerprint overlap: 0", size=5.6, weight="bold")

    d = ix(3)
    card(ax, d, 3.55, inner_w, 0.75)
    label(ax, d + 0.06, 4.08, "glm-5.3-shouyun", size=6.1, weight="bold")
    label(ax, d + 0.06, 3.86, "1,000 replies scored", size=5.7)
    label(ax, d + 0.06, 3.68, "985 kept", size=5.6, color=MUTED)

    stats = [
        ("901", "match as written", TEAL),
        ("84", "name repaired", ORANGE),
        ("15", "spectra differ", CORAL),
    ]
    y = 3.28
    for num, desc, color in stats:
        card(ax, d, y - 0.42, inner_w, 0.50)
        label(ax, d + 0.06, y - 0.16, num, size=8.0, weight="bold", color=mix(color, NAVY, 0.25))
        label(ax, d + 0.40, y - 0.16, desc, size=5.4)
        y -= 0.56
    label(ax, d, 1.05, "84 name repairs are in", size=5.4)
    label(ax, d, 0.87, "the training file.", size=5.4)
    label(ax, d, 0.69, "15 mismatches dropped.", size=5.4)

    e = ix(4)
    card(ax, e, 3.42, inner_w, 0.88, fc=WHITE, ec=mix(NAVY, WHITE, 0.45))
    label(ax, e + 0.06, 4.08, "No student scored", size=6.2, weight="bold")
    label(ax, e + 0.06, 3.86, "Format-only copies JSON", size=5.5)
    label(ax, e + 0.06, 3.68, "Visible reply: the result", size=5.5)
    label(ax, e + 0.06, 3.50, "The trace is not a target", size=5.5)

    card(ax, e, 2.05, inner_w, 1.28, fc=WHITE, ec=mix(NAVY, WHITE, 0.45))
    label(ax, e + 0.06, 3.12, "A rule, not a result", size=6.2, weight="bold")
    label(ax, e + 0.06, 2.90, "Pass@1 gap ≥ 10/120", size=5.7, weight="bold")
    label(ax, e + 0.06, 2.72, "matching samples", size=5.5, color=MUTED)
    label(ax, e + 0.06, 2.50, "Pass@5 gap ≥ 2/24", size=5.7, weight="bold")
    label(ax, e + 0.06, 2.32, "two designs, separate", size=5.5, color=MUTED)
    label(ax, e + 0.06, 2.14, "Not a significance test", size=5.4, color=MUTED)

    card(ax, e, 0.52, inner_w, 1.44, fc=mix(TEAL, WHITE, 0.82), ec=TEAL)
    label(ax, e + 0.06, 1.74, "Only measured row", size=6.2, weight="bold")
    label(ax, e + 0.06, 1.52, "Zero-shot Qwen3.5-9B", size=5.6)
    label(ax, e + 0.06, 1.32, "4/120 simulable", size=5.7)
    label(ax, e + 0.06, 1.14, "1/120 match", size=5.7, weight="bold", color=base.TEAL_INK)
    label(ax, e + 0.06, 0.90, "Same 24 problems", size=5.5)
    label(ax, e + 0.06, 0.72, "Without EF, n = 5", size=5.5)

    feet = ["Diagnose", "Build", "Verbalize", "Filter", "Not scored"]
    for x, (letter, _title, _subtitle, fill, accent), foot in zip(xs, PANELS, feet):
        rbox(
            ax,
            x,
            0.06,
            cw,
            0.26,
            fc=fill,
            ec=accent,
            lw=0.6,
            r=0.05,
            ls=(0, (2.0, 1.3)) if letter == "E" else "-",
        )
        label(ax, x + cw / 2, 0.19, foot, size=6.2, weight="bold", ha="center")

    audit_text(fig)
    base.save(fig, "framework")


def audit_text(fig) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.axes[0].texts if t.get_text().strip()]
    boxes = []
    for t in texts:
        bb = t.get_window_extent(renderer)
        boxes.append((t.get_text().replace("\n", " "), bb, t))
    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, bb_a, _ = boxes[i]
            b, bb_b, _ = boxes[j]
            if bb_a.overlaps(bb_b):
                hits.append((a[:36], b[:36]))
    # Flag text that leaves the 6.50 in canvas.
    ax = fig.axes[0]
    overflow = []
    for content, bb, t in boxes:
        inv = ax.transData.inverted()
        x0, y0 = inv.transform((bb.x0, bb.y0))
        x1, y1 = inv.transform((bb.x1, bb.y1))
        if x0 < -0.01 or x1 > 6.51 or y0 < -0.01 or y1 > 4.89:
            overflow.append((content[:40], round(x0, 2), round(x1, 2), round(y0, 2), round(y1, 2)))
    print(f"text overlaps: {len(hits)}")
    for pair in hits[:16]:
        print(" ", pair)
    print(f"canvas overflow: {len(overflow)}")
    for item in overflow[:12]:
        print(" ", item)


def main() -> None:
    base.setup_font()
    base.check_sources()
    draw()
    print("wrote framework.pdf")


if __name__ == "__main__":
    main()
