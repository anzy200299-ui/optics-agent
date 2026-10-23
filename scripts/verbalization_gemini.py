
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"

# Configure matplotlib settings
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'

# Create figure with exact dimensions
fig = plt.figure(figsize=(6.50, 4.20), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 6.50)
ax.set_ylim(0, 4.20)
ax.axis('off')

# Colors
c_bg = '#FFFFFF'
c_panel = '#EDF2F7'
c_primary = '#243B53'
c_accent1 = '#4E79A7'
c_accent2 = '#59A89C'
c_accent3 = '#F2B56B'
c_accent4 = '#D98282'

# Draw background
fig.patch.set_facecolor(c_bg)
ax.set_facecolor(c_bg)

# ==========================================
# LEFT STACK (Five rows)
# ==========================================
left_x = 0.12
left_w = 3.15
row_h = 0.68
row_gap = 0.06
top_y = 3.40

rows_data = [
    ("1. Index", "Ellipsis 280, comment 160, wrong model 280", "Ports 80, functional 120, graft 80"),
    ("2. Sample", "Widths {2, 3, 5, 6}; one SAX id changes (mzi vs mzm)"),
    ("3. Reject", "Port used twice, dangling bind, width 4 or 8", "8-QAM, 64-QAM, and the test WDM wavelengths"),
    ("4. SAX", "8 wavelengths in [1.51, 1.59] micrometers", r"Keep finite $|S|^{2}$. Not the 1,000-point test"),
    ("5. Write", "Prose is the prompt; JSON is held out"),
]

for i, row in enumerate(rows_data):
    title, *details = row
    y = top_y - i * (row_h + row_gap)
    
    # Row box
    rect = patches.FancyBboxPatch((left_x, y), left_w, row_h,
                                boxstyle="round,pad=0.01,rounding_size=0.04",
                                linewidth=0.8, edgecolor=c_primary, facecolor='#FFFFFF', zorder=2)
    ax.add_patch(rect)
    
    # Text inside row
    ax.text(left_x + 0.10, y + 0.46, title, fontsize=8, fontweight='bold', color=c_primary, zorder=3)
    for j, detail in enumerate(details):
        ax.text(left_x + 0.10, y + 0.26 - j * 0.16, detail, fontsize=6.5, color='#333333', zorder=3)

# ==========================================
# RIGHT PANEL
# ==========================================
right_x = 3.42
right_w = 2.96
right_y = 0.18
right_h = 3.90

panel_rect = patches.FancyBboxPatch((right_x, right_y), right_w, right_h,
                                   boxstyle="round,pad=0.01,rounding_size=0.06",
                                   linewidth=0.8, edgecolor='#CBD5E1', facecolor=c_panel, zorder=1)
ax.add_patch(panel_rect)

# 1. Horizontal stacked bar for families (280, 160, 280, 80, 120, 80 -> sum 1000)
counts = [280, 160, 280, 80, 120, 80]
colors = ['#F2B56B', '#4E79A7', '#59A89C', '#243B53', '#D98282', '#FFFFFF']
edges = ['none', 'none', 'none', 'none', 'none', '#243B53']
bar_y = 3.45
bar_h = 0.28
total = sum(counts)

current_x = right_x + 0.18
bar_w_total = right_w - 0.36

for count, color, edge in zip(counts, colors, edges):
    seg_w = (count / total) * bar_w_total
    seg = patches.Rectangle((current_x, bar_y), seg_w, bar_h, facecolor=color, edgecolor=edge, linewidth=0.6, zorder=2)
    ax.add_patch(seg)
    current_x += seg_w

# Legend for the stacked bar (2 columns inside the right panel)
legend_labels = ['Ellipsis (280)', 'Comment (160)', 'Wrong model (280)', 'Ports (80)', 'Functional (120)', 'Graft (80)']
leg_x_cols = [right_x + 0.18, right_x + 1.55]
leg_y_rows = [3.15, 2.95, 2.75]

for idx, (label, color) in enumerate(zip(legend_labels, colors)):
    col_idx = idx % 2
    row_idx = idx // 2
    lx = leg_x_cols[col_idx]
    ly = leg_y_rows[row_idx]
    
    swatch = patches.Rectangle((lx, ly), 0.10, 0.10, facecolor=color, edgecolor='none', zorder=2)
    ax.add_patch(swatch)
    ax.text(lx + 0.14, ly + 0.05, label, fontsize=6.4, color=c_primary, va='center', zorder=2)

# 2. Width ticks and bars (widths: 2, 3, 4, 5, 6, 8; values: 59, 76, 0, 70, 75, 0)
width_categories = [2, 3, 4, 5, 6, 8]
width_values = [59, 76, 0, 70, 75, 0]
max_w_val = 80 # scaling factor

w_start_y = 2.15
w_bar_h = 0.16
w_row_spacing = 0.25

ax.text(right_x + 0.18, w_start_y + 0.15, "Width distribution (n=280 items):", fontsize=6.8, fontweight='bold', color=c_primary, zorder=2)

for i, (w_cat, w_val) in enumerate(zip(width_categories, width_values)):
    row_y = w_start_y - (i + 1) * w_row_spacing
    
    # Label
    ax.text(right_x + 0.18, row_y + 0.04, f"w = {w_cat}:", fontsize=6.4, color=c_primary, fontweight='bold', zorder=2)
    
    if w_val > 0:
        bar_len = (w_val / max_w_val) * 1.30
        w_bar = patches.Rectangle((right_x + 0.70, row_y + 0.02), bar_len, w_bar_h, facecolor=c_accent1, edgecolor='none', zorder=2)
        ax.add_patch(w_bar)
        ax.text(right_x + 0.75 + bar_len, row_y + 0.04, f"{w_val}", fontsize=6.4, color=c_primary, zorder=2)
    else:
        ax.text(right_x + 0.70, row_y + 0.04, "held out (0)", fontsize=6.4, color='#718096', fontstyle='italic', zorder=2)

# 3. Footer inside the right panel
footer_text = "Weights not refit. Expert-fingerprint overlap: 0.\nSeed 20261020. No student trained."
ax.text(right_x + 0.18, 0.35, footer_text, fontsize=6.4, color=c_primary, linespacing=1.4, zorder=2)

# Save outputs without bbox_inches='tight' but exact pad_inches
plt.savefig(OUT / "verbalization_gemini.pdf", pad_inches=0.02, dpi=300)
plt.savefig(OUT / "verbalization_gemini.svg", pad_inches=0.02, dpi=300)
plt.savefig(OUT / "verbalization_gemini.png", pad_inches=0.02, dpi=300)

plt.close()
