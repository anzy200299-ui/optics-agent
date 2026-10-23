"""Vector figures for the PICBench failure-locus paper.

The three figures are the ones included from paper/main.tex.
Counts are the published Without-EF audit. When the score files are
present, main() checks the drawing constants against them and stops
if they disagree.

Outputs, next to the LaTeX sources:
  paper/figures/two_stage.{pdf,svg,png}
  paper/figures/verbalization.{pdf,svg,png}
  paper/figures/preregister.{pdf,svg,png}
PNG files are 300 dpi. PDF fonts are embedded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

# Palette specified for the paper figures.
NAVY = "#243B53"
BLUE = "#4E79A7"
TEAL = "#59A89C"
ORANGE = "#F2B56B"
CORAL = "#D98282"
GRAY = "#EDF2F7"
WHITE = "#FFFFFF"


def _rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def mix(a: str, b: str, t: float) -> str:
    aa, bb = _rgb(a), _rgb(b)
    rgb = tuple((1 - t) * x + t * y for x, y in zip(aa, bb))
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02X}" for c in rgb)


MUTED = mix(NAVY, WHITE, 0.38)
RULE = mix(NAVY, WHITE, 0.78)
ORANGE_LT = mix(ORANGE, WHITE, 0.42)
BLUE_LT = mix(BLUE, WHITE, 0.48)
CORAL_LT = mix(CORAL, WHITE, 0.42)
TEAL_SOFT = mix(TEAL, WHITE, 0.86)
BLUE_SOFT = mix(BLUE, WHITE, 0.90)
# Darker inks for small text. The display hues stay the specified colors.
TEAL_INK = mix(TEAL, NAVY, 0.38)
CORAL_INK = mix(CORAL, NAVY, 0.42)
AMBER_INK = mix(ORANGE, NAVY, 0.55)
# Second member of each locus pair, same hue, darker, so the stack stays readable.
AMBER_DEEP = mix(ORANGE, NAVY, 0.18)
BLUE_DEEP = mix(BLUE, NAVY, 0.30)
CORAL_DEEP = mix(CORAL, NAVY, 0.22)

# Published partition of the 120 stored netlists (Table 2 / Table 4).
# sax_or_schema = runtime + undefined model name + ports.
QWEN = {
    "extract": 7,
    "json": 72,
    "sax": 37,
    "sax_runtime": 25,
    "model_name": 9,
    "ports": 3,
    "wrong": 3,
    "match": 1,
}
DEEPSEEK = {
    "extract": 1,
    "json": 0,
    "sax": 44,
    "sax_runtime": 12,
    "model_name": 8,
    "ports": 24,
    "wrong": 37,
    "match": 38,
}

# Generator manifest and the scored teacher prefix (Tables 6 and 7).
LOCI = [
    ("Ellipsis", 280, ORANGE, NAVY, "Parallel chains"),
    ("Comment", 160, AMBER_DEEP, NAVY, "Distinct heater settings"),
    ("Wrong model", 280, BLUE, WHITE, "One SAX identifier"),
    ("Ports", 80, BLUE_DEEP, WHITE, "External bindings"),
    ("Functional", 120, CORAL, NAVY, "One-parameter move"),
    ("Graft", 80, CORAL_DEEP, WHITE, "Order not in the 24 tests"),
]
WIDTHS = {2: 59, 3: 76, 5: 70, 6: 75}
# Historical 175-reply prefix, still checked against failure_audit.json.
TEACHER_N = 175
TEACHER_MATCH = 141
TEACHER_NAME = 33
TEACHER_PARSE = 1
# Full 1,000-reply filter in logs/sft/picbench_glm53_visible_filter.json.
FILTER_N = 1000
FILTER_FUNC = 901
FILTER_REPAIRED = 84
FILTER_MISMATCH = 15
FILTER_KEPT = 985


def setup_font() -> None:
    for name in (
        "texgyreheros-regular.otf",
        "texgyreheros-bold.otf",
        "texgyreheros-italic.otf",
        "texgyreheros-bolditalic.otf",
    ):
        candidates = (
            Path("/usr/share/texmf/fonts/opentype/public/tex-gyre") / name,
            ROOT / "paper" / "fonts" / name,
            Path("/tmp/texgyre") / name,
        )
        path = next((item for item in candidates if item.exists()), None)
        if path is not None:
            fm.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": "TeX Gyre Heros",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "text.color": NAVY,
        }
    )


def canvas(w: float, h: float):
    fig = plt.figure(figsize=(w, h), dpi=120, facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def rbox(ax, x, y, w, h, *, fc=WHITE, ec=NAVY, lw=0.8, r=0.06, ls="-"):
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
            mutation_aspect=1,
        )
    )


def text(ax, x, y, s, *, size=8, weight="normal", color=NAVY, ha="left", va="center", style="normal", family=None, spacing=1.15):
    kwargs = {}
    if family:
        kwargs["fontfamily"] = family
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
        linespacing=spacing,
        zorder=4,
        **kwargs,
    )


def arrow(ax, x1, y1, x2, y2, *, color=NAVY, lw=0.8):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            joinstyle="miter",
            zorder=3,
        )
    )


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", facecolor=WHITE)
    fig.savefig(OUT / f"{stem}.svg", facecolor=WHITE)
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor=WHITE)
    plt.close(fig)


def hbar(ax, x, y, w, h, color) -> None:
    if w <= 0:
        return
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", zorder=2))


def track(ax, x, y, w, h) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=GRAY, edgecolor="none", zorder=1))


# ---------------------------------------------------------------------------
# Figure 1. Where the 120 replies stop.
# ---------------------------------------------------------------------------

def fig_outcomes() -> None:
    W, H = 6.50, 3.72
    fig, ax = canvas(W, H)

    text(
        ax,
        0.16,
        3.56,
        "A simulable netlist is not a spectrum match.",
        size=9.4,
        weight="bold",
        va="center",
    )
    text(
        ax,
        6.34,
        3.56,
        "Without EF   ·   24 × 5",
        size=7.4,
        color=MUTED,
        ha="right",
        va="center",
    )
    text(
        ax,
        0.16,
        3.36,
        "Bars partition 120 stored netlists.  Pass@1 and Pass@5 are syntax / function.",
        size=7.3,
        color=MUTED,
        va="center",
    )

    # Column geometry. One inch of track = 120 samples.
    lab_r = 1.72
    track_w = 1.78
    q_x = 1.86
    d_x = 4.28
    q_num = 3.70
    d_num = 6.12
    bar_h = 0.20

    def col_head(cx, name, p1, p5):
        text(ax, cx, 3.12, name, size=8.3, weight="bold", ha="center", va="center")
        text(ax, cx, 2.92, f"Pass@1    {p1}", size=7.3, color=MUTED, ha="center", va="center")
        text(ax, cx, 2.76, f"Pass@5    {p5}", size=7.3, color=MUTED, ha="center", va="center")

    col_head(q_x + track_w / 2, "Qwen3.5-9B", "3.33%  /  0.83%", "16.67%  /  4.17%")
    col_head(d_x + track_w / 2, "DeepSeek-V4-Pro", "62.50%  /  31.67%", "95.83%  /  62.50%")

    def section(y, label):
        text(ax, 0.16, y, label, size=7.3, weight="bold", color=MUTED, va="center")
        ax.plot([1.86, 6.34], [y, y], color=RULE, lw=0.6, zorder=1)

    def row(y, label, color, qn, dn):
        text(ax, lab_r, y + bar_h / 2, label, size=7.8, ha="right", va="center")
        track(ax, q_x, y, track_w, bar_h)
        track(ax, d_x, y, track_w, bar_h)
        hbar(ax, q_x, y, track_w * qn / 120, bar_h, color)
        hbar(ax, d_x, y, track_w * dn / 120, bar_h, color)
        q_ink = TEAL_INK if color == TEAL else CORAL_INK if color == CORAL else AMBER_INK if color == ORANGE else NAVY
        d_ink = q_ink
        text(ax, q_num, y + bar_h / 2, str(qn), size=8.4, weight="bold", color=q_ink, ha="left", va="center")
        text(ax, d_num, y + bar_h / 2, str(dn), size=8.4, weight="bold", color=d_ink, ha="left", va="center")

    section(2.50, "SAX does not return a spectrum")
    row(2.16, "JSON not loaded", ORANGE, QWEN["extract"] + QWEN["json"], DEEPSEEK["extract"] + DEEPSEEK["json"])
    row(1.84, "SAX or schema rejects", BLUE, QWEN["sax"], DEEPSEEK["sax"])

    section(1.58, "SAX returns a spectrum")
    row(1.24, "Spectrum differs", CORAL, QWEN["wrong"], DEEPSEEK["wrong"])
    row(0.92, "Spectrum matches", TEAL, QWEN["match"], DEEPSEEK["match"])

    text(ax, q_x, 0.74, "0", size=7.0, color=MUTED, ha="center")
    text(ax, q_x + track_w, 0.74, "120", size=7.0, color=MUTED, ha="center")
    text(ax, d_x, 0.74, "0", size=7.0, color=MUTED, ha="center")
    text(ax, d_x + track_w, 0.74, "120", size=7.0, color=MUTED, ha="center")
    text(ax, q_x + track_w / 2, 0.74, "samples", size=7.0, color=MUTED, ha="center")
    text(ax, d_x + track_w / 2, 0.74, "samples", size=7.0, color=MUTED, ha="center")

    text(
        ax,
        0.16,
        0.50,
        "Conditional on a spectrum: Qwen 1/4 = 25%, DeepSeek 38/75 = 50.7%. Syntax Pass@k includes matches; the bars are exclusive.",
        size=7.25,
        color=NAVY,
        va="center",
    )
    text(
        ax,
        0.16,
        0.32,
        "JSON not loaded: Qwen 7 with no result span + 72 rejected by json.loads; DeepSeek 1 + 0.",
        size=7.2,
        color=MUTED,
        va="center",
    )
    text(
        ax,
        0.16,
        0.15,
        "SAX or schema rejects, Qwen / DeepSeek: runtime 25/12,  model name 9/8,  ports 3/24.",
        size=7.2,
        color=MUTED,
        va="center",
    )

    save(fig, "two_stage")


# ---------------------------------------------------------------------------
# Figure 2. Failure-locus verbalization.
# ---------------------------------------------------------------------------

def fig_verbalization() -> None:
    W, H = 6.50, 4.05
    fig, ax = canvas(W, H)

    text(ax, 0.16, 3.86, "The circuit is checked before the prose is written.", size=9.4, weight="bold", va="center")
    text(ax, 6.34, 3.86, "JSON stays out of the prompt", size=7.4, color=MUTED, ha="right", va="center")

    steps = [
        (BLUE, BLUE, "1   Index", "9B text-heuristic labels.\nNot a PICBench device family."),
        (BLUE, BLUE, "2   Build", "One legal SAX graph.\nOne free variable inside the locus."),
        (TEAL, TEAL_INK, "3   Keep", "Finite |S|² on 8 wavelengths.\nNot the 1,000-point equality test."),
        (TEAL, TEAL_INK, "4   Verbalize", "Prose is the prompt.\nJSON is stored as the reference."),
    ]
    bx, bw, bh, gap = 0.16, 2.30, 0.74, 0.12
    top = 3.62
    bottoms = []
    for i, (accent, title_c, title, body) in enumerate(steps):
        y = top - (i + 1) * bh - i * gap
        rbox(ax, bx, y, bw, bh, fc=WHITE, ec=RULE, lw=0.7, r=0.05)
        ax.add_patch(Rectangle((bx + 0.06, y + 0.09), 0.030, bh - 0.18, facecolor=accent, edgecolor="none", zorder=3))
        text(ax, bx + 0.16, y + bh - 0.18, title, size=8.0, weight="bold", color=title_c, va="center")
        text(ax, bx + 0.16, y + 0.26, body, size=7.4, color=NAVY, va="center", spacing=1.28)
        bottoms.append(y)
        if i:
            arrow(
                ax,
                bx + 0.34,
                bottoms[i - 1] - 0.012,
                bx + 0.34,
                y + bh + 0.012,
                color=mix(NAVY, WHITE, 0.28),
            )

    rx, rw = 2.66, 3.68
    text(ax, rx, 3.46, "1,000 SAX-checked questions", size=8.2, weight="bold", va="center")
    text(ax, rx + rw, 3.46, "seed 20261020", size=7.2, color=MUTED, ha="right", va="center")

    bar_y, bar_h = 3.12, 0.22
    gap_w = 0.014
    total = sum(n for _, n, _, _, _ in LOCI)
    usable = rw - gap_w * (len(LOCI) - 1)
    cursor = rx
    for name, n, color, ink, _sub in LOCI:
        w = usable * n / total
        hbar(ax, cursor, bar_y, w, bar_h, color)
        if w >= 0.55:
            text(ax, cursor + w / 2, bar_y + bar_h / 2, str(n), size=7.3, weight="bold", color=ink, ha="center", va="center")
        cursor += w + gap_w

    for i, (name, n, color, _ink, sub) in enumerate(LOCI):
        col, row = i % 3, i // 3
        x = rx + col * 1.24
        y = 2.90 - row * 0.34
        ax.add_patch(Rectangle((x, y - 0.05), 0.11, 0.11, facecolor=color, edgecolor="none", zorder=3))
        text(ax, x + 0.17, y + 0.055, f"{name}  {n}", size=7.3, weight="bold", va="center")
        text(ax, x + 0.17, y - 0.085, sub, size=7.0, color=MUTED, va="center")

    text(ax, rx, 2.22, "Items that carry a width, n = 280.   Widths 4 and 8 are held out.", size=7.3, weight="bold", va="center")
    order = [2, 3, 4, 5, 6, 8]
    slot = rw / len(order)
    base, max_h, max_c = 1.58, 0.40, 76
    ax.plot([rx, rx + rw], [base, base], color=RULE, lw=0.6, zorder=1)
    for i, width in enumerate(order):
        cx = rx + (i + 0.5) * slot
        count = WIDTHS.get(width, 0)
        if count:
            bh = max_h * count / max_c
            hbar(ax, cx - 0.12, base, 0.24, bh, BLUE)
            text(ax, cx, base + bh + 0.025, str(count), size=7.2, ha="center", va="bottom")
            text(ax, cx, base - 0.10, str(width), size=7.3, ha="center", va="top")
        else:
            text(ax, cx, base + 0.05, "held out", size=6.9, color=AMBER_INK, ha="center", va="bottom")
            text(ax, cx, base - 0.10, str(width), size=7.3, weight="bold", color=AMBER_INK, ha="center", va="top")

    panel_y, panel_h = 0.12, 1.16
    cw = 1.58
    rbox(ax, rx, panel_y, cw, panel_h, fc=BLUE_SOFT, ec=mix(BLUE, WHITE, 0.50), lw=0.6, r=0.05)
    rbox(ax, rx + cw + 0.12, panel_y, rw - cw - 0.12, panel_h, fc=GRAY, ec=mix(NAVY, WHITE, 0.82), lw=0.6, r=0.05)

    text(ax, rx + 0.10, panel_y + panel_h - 0.14, "Contrast set", size=7.6, weight="bold", va="center")
    text(ax, rx + 0.10, panel_y + panel_h - 0.30, "Same wiring. One SAX identifier.", size=7.0, color=MUTED, va="center")
    _circuit(ax, rx + 0.16, panel_y + 0.58, "mzi", ports=True)
    _circuit(ax, rx + 0.16, panel_y + 0.24, "mzm", ports=False)

    hx = rx + cw + 0.24
    text(ax, hx, panel_y + panel_h - 0.15, "Held out", size=7.6, weight="bold", va="center")
    held = [
        "24 problems and expert JSON",
        "Teacher answers on those 24",
        "Widths 4 and 8; 8-QAM, 64-QAM, WDM",
        "Expert-fingerprint overlap: 0",
    ]
    yy = panel_y + panel_h - 0.34
    for item in held:
        ax.add_patch(Circle((hx + 0.025, yy), 0.018, facecolor=NAVY, edgecolor="none", zorder=3))
        text(ax, hx + 0.09, yy, item, size=7.1, va="center")
        yy -= 0.145
    text(
        ax,
        hx,
        panel_y + 0.16,
        "9B labels on these 24 choose the families.\nWeights were not refit after the re-audit.",
        size=7.0,
        color=MUTED,
        va="center",
        spacing=1.22,
    )

    save(fig, "verbalization")


def _circuit(ax, x, y, ident: str, ports: bool = False) -> None:
    ax.add_patch(Circle((x + 0.06, y), 0.026, facecolor=NAVY, edgecolor="none", zorder=3))
    ax.plot([x + 0.09, x + 0.24], [y, y], color=NAVY, lw=0.8, zorder=2)
    rbox(ax, x + 0.24, y - 0.085, 0.42, 0.17, fc=WHITE, ec=BLUE, lw=0.7, r=0.04)
    text(ax, x + 0.45, y, ident, size=7.2, weight="bold", ha="center", va="center", family="DejaVu Sans Mono", color=BLUE)
    ax.plot([x + 0.66, x + 0.81], [y, y], color=NAVY, lw=0.8, zorder=2)
    ax.add_patch(Circle((x + 0.84, y), 0.026, facecolor=NAVY, edgecolor="none", zorder=3))
    if ports:
        text(ax, x + 0.06, y + 0.055, "I1", size=6.6, color=MUTED, ha="center", va="bottom")
        text(ax, x + 0.84, y + 0.055, "O1", size=6.6, color=MUTED, ha="center", va="bottom")


# ---------------------------------------------------------------------------
# Figure 3. The reference-JSON arm is still prospective. Epoch 8 is scored.
# ---------------------------------------------------------------------------

def fig_prereg() -> None:
    W, H = 6.50, 4.22
    fig, ax = canvas(W, H)

    text(ax, 0.16, 4.04, "Epoch 8 is scored. The reference-JSON arm is not.", size=9.2, weight="bold", va="center")
    text(ax, 6.34, 4.04, "Without EF  ·  same judge", size=7.4, color=MUTED, ha="right", va="center")

    cols = [
        (0.14, 1.46),
        (1.76, 1.62),
        (3.54, 1.40),
        (5.10, 1.26),
    ]
    top, tall = 2.68, 1.18

    x, w = cols[0]
    rbox(ax, x, top, w, tall, fc=WHITE, ec=NAVY, lw=0.8, r=0.05)
    text(ax, x + 0.10, top + tall - 0.16, "Measured pool", size=7.3, weight="bold", color=TEAL_INK, va="center")
    text(ax, x + 0.10, top + 0.62, "1,000 SAX-checked\nquestions\nJSON never in the prompt", size=7.5, va="center", spacing=1.28)
    text(ax, x + 0.10, top + 0.14, "Seed 20261020", size=7.1, color=MUTED, va="center")

    x, w = cols[1]
    rbox(ax, x, top, w, tall, fc=WHITE, ec=NAVY, lw=0.8, r=0.05)
    text(ax, x + 0.10, top + tall - 0.16, "All 1,000 scored", size=7.3, weight="bold", color=TEAL_INK, va="center")
    text(ax, x + 0.10, top + 0.80, "901 match as written", size=7.4, color=TEAL_INK, weight="bold", va="center")
    text(ax, x + 0.10, top + 0.62, "84 after a name repair", size=7.4, color=AMBER_INK, va="center")
    text(ax, x + 0.10, top + 0.44, "15 spectra differ", size=7.4, color=CORAL_INK, va="center")
    text(ax, x + 0.10, top + 0.26, "985 kept", size=7.4, weight="bold", va="center")
    text(ax, x + 0.10, top + 0.10, "Raw files are unchanged", size=7.0, color=MUTED, va="center")

    x, w = cols[2]
    h2 = (tall - 0.10) / 2
    students = (
        (top + h2 + 0.10, "Not scored", "Reference JSON", "Same 2% rule, no score", True),
        (top, "Epoch 8 scored", "Teacher answer", "27 in SAX, 4 match", False),
    )
    for y, status, title, body, dashed in students:
        if dashed:
            rbox(ax, x, y, w, h2, fc=GRAY, ec=BLUE, lw=0.8, r=0.05, ls=(0, (2.4, 1.5)))
            status_color = BLUE
        else:
            rbox(ax, x, y, w, h2, fc=WHITE, ec=NAVY, lw=0.8, r=0.05)
            status_color = TEAL_INK
        text(ax, x + 0.08, y + h2 - 0.14, status, size=7.0, weight="bold", color=status_color, va="center")
        text(ax, x + 0.08, y + 0.28, title, size=7.4, weight="bold", va="center")
        text(ax, x + 0.08, y + 0.12, body, size=7.0, color=MUTED, va="center")

    x, w = cols[3]
    rbox(ax, x, top + 0.54, w, tall - 0.54, fc=WHITE, ec=NAVY, lw=0.8, r=0.05)
    text(ax, x + 0.08, top + tall - 0.14, "Teacher, epoch 8", size=7.0, weight="bold", color=TEAL_INK, va="center")
    text(ax, x + 0.08, top + 0.84, "27/120 simulable", size=7.15, va="center")
    text(ax, x + 0.08, top + 0.68, "4/120 match", size=7.15, weight="bold", va="center")
    rbox(ax, x, top, w, 0.48, fc=WHITE, ec=NAVY, lw=0.7, r=0.04)
    text(ax, x + 0.08, top + 0.36, "Zero-shot, measured", size=6.9, weight="bold", color=TEAL_INK, va="center")
    text(ax, x + 0.08, top + 0.22, "4/120 simulable", size=7.15, va="center")
    text(ax, x + 0.08, top + 0.08, "1/120 match", size=7.15, weight="bold", va="center")

    mid = top + tall / 2
    arrow(ax, 1.60, mid, 1.75, mid)
    arrow(ax, 3.38, mid + 0.30, 3.53, mid + 0.30)
    arrow(ax, 3.38, mid - 0.30, 3.53, mid - 0.30)
    arrow(ax, 4.94, top + 0.54 + (tall - 0.54) / 2, 5.09, top + 0.54 + (tall - 0.54) / 2)

    text(
        ax,
        0.16,
        2.48,
        "Both students would use the 985 kept replies. Name repairs are already in that file. The 15 mismatches are dropped.",
        size=7.2,
        color=MUTED,
        va="center",
    )

    card_y, card_h, card_w = 0.86, 1.42, 3.02
    _decision_card(
        ax, 0.14, card_y, card_w, card_h,
        "Functional Pass@1", "≥  10/120", "additional matching samples", "A sample rate. Not two designs.",
    )
    _decision_card(
        ax, 0.14 + card_w + 0.18, card_y, card_w, card_h,
        "Functional Pass@5", "≥  2/24", "additional designs", "What “two designs” means.",
    )

    rbox(ax, 0.14, 0.12, 6.22, 0.62, fc=GRAY, ec="none", lw=0, r=0.04)
    text(ax, 0.28, 0.56, "Not a forecast.", size=7.4, weight="bold", va="center")
    text(
        ax,
        0.28,
        0.32,
        "If all 79 unloadable Qwen replies became simulation-valid misses, simulable netlists would be 83/120\nand spectrum matches would stay 1/120. A syntax-only gain clears neither bar. Not a significance test.",
        size=7.15,
        color=NAVY,
        va="center",
        spacing=1.28,
    )

    save(fig, "preregister")


def _decision_card(ax, x, y, w, h, kicker, value, unit, note) -> None:
    rbox(ax, x, y, w, h, fc=WHITE, ec=NAVY, lw=0.85, r=0.05)
    text(ax, x + w / 2, y + h - 0.22, kicker, size=8.0, weight="bold", ha="center", va="center")
    text(ax, x + w / 2, y + h - 0.62, value, size=16, weight="bold", ha="center", va="center")
    text(ax, x + w / 2, y + 0.40, unit, size=7.8, ha="center", va="center")
    text(ax, x + w / 2, y + 0.18, note, size=7.2, color=MUTED, ha="center", va="center")


def _expect(name: str, got, want) -> None:
    if got != want:
        raise SystemExit(f"{name}: log has {got}, figure uses {want}")


def check_sources() -> None:
    audit_path = ROOT / "logs/picbench/failure_audit.json"
    manifest_path = ROOT / "data/weakness_questions/manifest.json"
    overlap_path = ROOT / "logs/picbench/structure_overlap.json"
    if not (audit_path.exists() and manifest_path.exists() and overlap_path.exists()):
        print("source files not all present; drew the published constants")
        return

    audit = json.loads(audit_path.read_text())
    for key, spec in (("qwen35-9b", QWEN), ("deepseek-v4-pro", DEEPSEEK)):
        off = audit[key]["official"]
        sax = off["sax_rejected"] + off["model_name_rejected"] + off["port_rejected"]
        _expect(f"{key}.extract", off["extraction_failure"], spec["extract"])
        _expect(f"{key}.json", off.get("json_parse_failure", 0), spec["json"])
        _expect(f"{key}.sax", sax, spec["sax"])
        _expect(f"{key}.runtime", off["sax_rejected"], spec["sax_runtime"])
        _expect(f"{key}.name", off["model_name_rejected"], spec["model_name"])
        _expect(f"{key}.ports", off["port_rejected"], spec["ports"])
        _expect(f"{key}.wrong", off["simulation_valid_spectrum_differs"], spec["wrong"])
        _expect(f"{key}.match", off["spectrum_match"], spec["match"])
        total = spec["extract"] + spec["json"] + spec["sax"] + spec["wrong"] + spec["match"]
        _expect(f"{key}.total", total, 120)

    prefix = audit["teacher_prefix"]
    _expect("teacher.n", prefix["scored_prefix"], TEACHER_N)
    _expect("teacher.name", prefix["syntax_rows"], TEACHER_NAME)
    _expect("teacher.account", TEACHER_MATCH + TEACHER_NAME + TEACHER_PARSE, TEACHER_N)

    filt_path = ROOT / "logs/sft/picbench_glm53_visible_filter.json"
    if filt_path.exists():
        filt = json.loads(filt_path.read_text())
        _expect("filter.n", filt["n_scored"], FILTER_N)
        _expect("filter.func", filt["counts"]["func"], FILTER_FUNC)
        _expect("filter.repaired", filt["counts"]["func_repaired"], FILTER_REPAIRED)
        _expect("filter.mismatch", filt["counts"]["spectrum_mismatch"], FILTER_MISMATCH)
        _expect("filter.kept", filt["n_kept"], FILTER_KEPT)
        _expect(
            "filter.sum",
            FILTER_FUNC + FILTER_REPAIRED + FILTER_MISMATCH,
            FILTER_N,
        )

    manifest = json.loads(manifest_path.read_text())
    _expect("manifest.count", manifest["count"], 1000)
    _expect("manifest.seed", manifest["seed"], 20261020)
    _expect("manifest.sax", manifest["sax_checked"], 1000)
    for name, n, *_ in LOCI:
        key = {
            "Ellipsis": "ellipsis",
            "Comment": "comment",
            "Wrong model": "wrong_model",
            "Ports": "wrong_ports",
            "Functional": "functional",
            "Graft": "graft",
        }[name]
        _expect(f"locus.{key}", manifest["by_locus"][key], n)

    overlap = json.loads(overlap_path.read_text())
    _expect("fingerprints", overlap["exact_fingerprint_matches"], [])
    for width, count in WIDTHS.items():
        _expect(f"width.{width}", overlap["train_widths"][str(width)], count)


def main() -> None:
    setup_font()
    check_sources()
    fig_outcomes()
    fig_verbalization()
    fig_prereg()
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()


