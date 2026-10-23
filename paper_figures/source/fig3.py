"""Figure 3. Executability moved. The functional bar was not cleared.

Hand-drawn cards: the teacher filter on top, three measured student rows,
and a footer that keeps the pre-set bars honest.

Teacher filter: 901 as stored, 84 after an identity model-name repair, 15 dropped.
Student counts are out of 120. Reference-JSON epoch 8 is 31 simulable and 0 matches.
Teacher-answer epoch 8 is 27 and 4. The functional gap is 4/120, below 10/120.
The design-count gap is 4/24, above the separate 2/24 bar.
Neither bar is a significance test.
"""

from __future__ import annotations

from style import (
    AMBER_SOFT,
    BLUE,
    CORAL,
    GRAY,
    GRAY_SOFT,
    MUTED,
    NAVY,
    TEAL,
    WHITE,
    canvas,
    rect,
    rounded,
    save,
    setup,
    text,
)

W, H = 7.0, 4.7


def filter_card(ax) -> None:
    x, y, w, h = 0.16, 3.36, W - 0.32, 1.20
    rounded(ax, x, y, w, h, AMBER_SOFT, ec=NAVY, lw=1.0, radius=0.08)
    text(ax, x + 0.22, y + h - 0.24, "Teacher filter", fontsize=9.4, fontweight="bold")
    text(ax, x + 1.55, y + h - 0.24, "1,000-point |S|² equality on the synthetic references",
         fontsize=6.8, color=MUTED)

    parts = (
        (901, TEAL, None, WHITE),
        (84, BLUE, "///", WHITE),
        (15, CORAL, "xx", NAVY),
    )
    x0, by, bh, width = x + 0.22, y + 0.52, 0.30, 4.72
    scale = width / 1000.0
    cursor = x0
    for count, fc, hatch, ink in parts:
        seg = count * scale
        rect(ax, cursor, by, seg, bh, fc, ec=WHITE, lw=0.6, hatch=hatch, z=4)
        if seg > 0.55:
            text(ax, cursor + seg / 2, by + bh / 2, str(count), ha="center", va="center",
                 fontsize=8.4, color=ink, fontweight="bold", zorder=5)
        cursor += seg
    # 15 is a sliver: label it just right of the bar end.
    text(ax, x0 + width + 0.05, by + bh / 2, "15", ha="left", va="center",
         fontsize=7.2, color=CORAL, fontweight="bold")
    # legend under the bar
    labels = (
        (0.22, TEAL, None, "901 as stored"),
        (1.85, BLUE, "///", "84 repaired"),
        (3.35, CORAL, "xx", "15 dropped"),
    )
    for dx, fc, hatch, label in labels:
        rect(ax, x + dx, y + 0.22, 0.16, 0.11, fc, ec=NAVY, lw=0.4, hatch=hatch, z=4)
        text(ax, x + dx + 0.24, y + 0.275, label, fontsize=6.9)

    rounded(ax, x + 5.30, y + 0.36, 1.36, 0.56, WHITE, ec=TEAL, lw=1.1, radius=0.07)
    text(ax, x + 5.98, y + 0.72, "985 kept", ha="center", fontsize=9.0, fontweight="bold", color=TEAL)
    text(ax, x + 5.98, y + 0.50, "for training", ha="center", fontsize=6.8, color=MUTED)


def lollipop(ax, x0, y, value, scale, color, label) -> None:
    x = x0 + value * scale
    ax.plot([x0, x], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=4)
    ax.plot(x, y, marker="o", ms=8, color=color, markeredgecolor=NAVY, markeredgewidth=0.6, zorder=5)
    text(ax, x + 0.10, y, label, fontsize=8.0, va="center", color=NAVY, fontweight="bold")


def students_card(ax) -> None:
    x, y, w, h = 0.16, 1.24, W - 0.32, 2.00
    rounded(ax, x, y, w, h, WHITE, ec=NAVY, lw=1.0, radius=0.08)
    text(ax, x + 0.22, y + h - 0.24, "PICBench evaluation, 120 samples", fontsize=9.4, fontweight="bold")
    text(ax, x + w - 0.22, y + h - 0.24, "All three rows are measured.",
         fontsize=7.0, color=MUTED, ha="right")

    names = ("Zero-shot 9B", "Teacher-answer, epoch 8", "Reference-JSON, epoch 8")
    ys = (y + h - 0.68, y + h - 1.04, y + h - 1.42)
    for name, yy in zip(names, ys):
        text(ax, x + 0.22, yy, name, fontsize=7.6, fontweight="bold", color=NAVY)

    panels = (
        (x + 2.55, "Simulable / 120", BLUE, (4, 27, 31)),
        (x + 4.72, "Matching / 120", TEAL, (1, 4, 0)),
    )
    scale = 1.52 / 40.0
    for px, title, color, values in panels:
        text(ax, px, y + h - 0.46, title, fontsize=7.8, fontweight="bold", color=color)
        x1 = px + 40 * scale
        ax.plot([px, x1], [y + 0.30, y + 0.30], color=GRAY, lw=0.8, zorder=2)
        for tick in (0, 20, 40):
            tx = px + tick * scale
            ax.plot([tx, tx], [y + 0.30, y + 0.25], color=GRAY, lw=0.8)
            text(ax, tx, y + 0.17, str(tick), ha="center", va="top", fontsize=6.6, color=MUTED)
        for yy, value in zip(ys, values):
            lollipop(ax, px, yy, value, scale, color, str(value))


def boundary(ax) -> None:
    rounded(ax, 0.16, 0.10, W - 0.32, 1.02, GRAY_SOFT, ec="none", lw=0, radius=0.07)
    text(ax, 0.32, 0.90, "Teacher-answer minus reference-JSON, epoch 8",
         fontsize=8.0, fontweight="bold")
    text(ax, 0.32, 0.66,
         "Sample accuracy: +4/120 matches; pre-specified threshold +10/120.",
         fontsize=7.4, color=NAVY)
    text(ax, 0.32, 0.46,
         "Design coverage: +4/24 designs; pre-specified threshold +2/24.",
         fontsize=7.4, color=NAVY)
    text(ax, 0.32, 0.26,
         "Thresholds concern effect size. Checkpoint sensitivity is reported separately.",
         fontsize=7.0, color=MUTED)


def build() -> None:
    fig, ax = canvas(W, H)
    filter_card(ax)
    students_card(ax)
    boundary(ax)
    save(fig, "fig3_teacher_sft")


if __name__ == "__main__":
    setup()
    build()
