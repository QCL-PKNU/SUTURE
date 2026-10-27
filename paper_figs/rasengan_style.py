"""rasengan_style — shared plotting style for the manuscript figures.

Extracted from rasengan/reproduce/figure_{9,10,12}/only_plot.ipynb:
  - Times New Roman, pdf/ps fonttype 42 (TrueType, ASPLOS-safe), font 60
    on a 22x12-inch canvas (scaled proportionally per figure)
  - axes.linewidth 5, thick lines (4), big black-edged filled markers
  - dashed y-grid (linestyle '--', linewidth 1.5, axis='y')
  - palette: navy #384B70, green #6A9C89, red #B8001F, cream #FFF5E4;
    muted tints #8d97a9/#a9bfb7/#c6767f for stacked halves; hatch '/'
  - bars: black edge, thick linewidth; legends frameless, top-anchored
    multi-column (mode='expand')

Grayscale safety added on top (not in the originals): every multi-series
plot ALSO varies linestyle/marker/hatch, and palette luminances are
ordered (navy 0.28 < red 0.33 < green 0.55 < cream 0.95).

Usage:
    from rasengan_style import figure, ygrid, line_kw, bar_kw, save_fig
    fig, ax, s = figure(scale=0.5)          # 11x6in, proportional fonts
    ...
    save_fig(fig, "out", "fig_e1")          # fig_e1.pdf + fig_e1_600dpi.png
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

PAL = {
    "navy":   "#384B70",
    "green":  "#6A9C89",
    "red":    "#B8001F",
    "cream":  "#FFF5E4",
    "orange": "#E08A2E",
    "gray":   "#8d97a9",
    "greent": "#a9bfb7",
    "redt":   "#c6767f",
}
ARM_COLOR = {
    "purify": PAL["navy"], "purify_restart": PAL["gray"], "purify_last": "#5a6b8f",
    "rec_synd": PAL["red"], "synd": PAL["red"], "repair": PAL["red"], "repair_synd": PAL["red"],
    "rec_nearest": PAL["green"], "rec_uniform": PAL["orange"],
    "junk_synd": "#333333", "junk": "#333333", "feas_control": "#b8a26a",
}
ARM_LABEL = {
    "purify": "Rasengan (purify)", "purify_restart": "purify+restart",
    "purify_last": "purify+last", "rec_synd": "SUTURE (ours)", "synd": "SUTURE (ours)",
    "repair": "SUTURE (ours)", "repair_synd": "SUTURE (ours)",
    "rec_nearest": "coset-uniform", "rec_uniform": "uniform reinjection",
    "junk_synd": "SUTURE-on-junk (A0)", "feas_control": "feasible control",
}
LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
MK = ["o", "s", "^", "D", "v", "P", "X"]
HATCH = ["", "/", "\\\\", "xx", "..", "//"]

# Role-mapped size ladder, in the notebooks' 22x12-canvas units (divide by
# 22*wratio and multiply by the target width in inches to get on-page pt).
# NOTE: the reproduce notebooks themselves set legends SMALLER than axis
# labels (figure_10: legends 35/50, suptitle 48, axis labels 60) -- measured
# on the rendered PNGs too (legend 'Theorem 1' 24px vs x-label 26px). This
# ladder deliberately INVERTS that at our column width, per user request:
# the legend is what a reader must resolve at print size, the axis label is
# redundant context. Ladder (on-page pt at column width):
#   title 9.1 > legend 8.5 > axis label 7.5 > tick 7.0
BASE = dict(figsize=(22, 12), font=60, axlw=5, linelw=4, mks=22, mklw=3,
            grid_lw=1.5, legfs=64, tickfs=52, labelfs=56, titlefs=68)


def apply_style(scale=0.5):
    s = scale
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                       "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "serif",
        "font.size": BASE["font"] * s,
        "axes.linewidth": BASE["axlw"] * s,
        "axes.labelsize": BASE["labelfs"] * s,
        "axes.titlesize": BASE["titlefs"] * s,
        "xtick.labelsize": BASE["tickfs"] * s,
        "ytick.labelsize": BASE["tickfs"] * s,
        "legend.fontsize": BASE["legfs"] * s,
        "lines.linewidth": BASE["linelw"] * s,
        "lines.markersize": BASE["mks"] * s,
        "lines.markeredgewidth": BASE["mklw"] * s,
        "xtick.major.width": BASE["axlw"] * s * 0.8,
        "ytick.major.width": BASE["axlw"] * s * 0.8,
        "xtick.major.size": 10 * s,
        "ytick.major.size": 10 * s,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
    })
    return s


def figure(scale=0.5, ncols=1, nrows=1, wratio=1.0):
    s = apply_style(scale)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22 * s * wratio, 12 * s))
    return fig, axes, s


def ygrid(ax, s=0.5):
    ax.grid(True, linestyle="--", linewidth=BASE["grid_lw"] * s * 1.4,
            axis="y", color="#999999", alpha=0.8)
    ax.set_axisbelow(True)


def bar_kw(i, s=0.5, palette=None):
    cols = palette or [PAL["navy"], PAL["green"], PAL["red"], PAL["cream"]]
    return dict(color=cols[i % len(cols)], edgecolor="black",
                linewidth=BASE["axlw"] * s * 0.55, hatch=HATCH[i % len(HATCH)])


def line_kw(i, color=None):
    c = color or [PAL["navy"], PAL["red"], PAL["green"], PAL["orange"]][i % 4]
    return dict(color=c, linestyle=LS[i % len(LS)], marker=MK[i % len(MK)],
                markerfacecolor=c, markeredgecolor="black")


def top_legend(fig, ncol, y=0.98):
    # NOTE: no mode="expand" -- expand distributes columns over the full band
    # width regardless of label length, so long labels overprint their
    # neighbors (measured on fig_e1/fig_e2). Natural column packing instead.
    return fig.legend(loc="lower center", ncol=ncol, frameon=False,
                      bbox_to_anchor=(0.5, y), borderaxespad=0,
                      columnspacing=0.9, handletextpad=0.35,
                      labelspacing=0.25)


def save_fig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f"{name}.pdf")
    png = os.path.join(outdir, f"{name}_600dpi.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    print(f"saved {pdf} + {png}")
    return pdf, png


def check_overlaps(fig):
    # Force a fresh draw at the figure's own dpi before measuring. Without
    # this, a preceding savefig(dpi=600) leaves some text extents cached in
    # 600-dpi units while others re-measure at figure dpi -- mixed coordinate
    # systems that produce FALSE overlap flags (observed on fig_e2: panel (b)
    # ylabel at x=4173 on a 1254-px canvas).
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    ov = [(a.get_text()[:16], b.get_text()[:16])
          for i, (a, ba) in enumerate(texts)
          for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    ov = [o for o in ov if not any(ch.isdigit() for ch in o[0] + o[1])]
    if ov:
        print("TEXT OVERLAPS:", ov[:6])
    return ov


def arm_label(arm, variant=None):
    """Arm label with the decoder-variant subscript the manuscript requires.

    variant="C": constraint-only decoder (hard-edge workloads, KV, cross-family)
    variant="G": soft-edge adjacency-aware specialization
    The subscript is a property of the WORKLOAD, not the arm name -- the same
    `rec_synd` column is SUTURE_C on a hard-edge pack and SUTURE_G on a soft-edge
    one -- so each figure passes the variant its data came from.
    """
    base = ARM_LABEL.get(arm, arm)
    if variant and "SUTURE" in base and "$_" not in base:
        base = base.replace("SUTURE", f"SUTURE$_{variant}$")
    return base


# Column-width figures (single-column acmart: 3.343in text, 241.15pt).
# On-page font = 60*W_target/(22*wratio) and is INDEPENDENT of scale, so a
# figure moved from \begin{figure*} to \begin{figure} must have wratio
# re-derived or its text renders ~2x too large. COL_WRATIO puts a 2-panel
# strip at 8.5pt, matching acmart's caption size.
COL_W = 3.343          # inches, \columnwidth
TEXT_W = 7.02          # inches, \textwidth
COL_WRATIO = 1.14      # 2-panel single column -> 8.0pt on page


def col_figure(ncols=2, nrows=1, wratio=COL_WRATIO, scale=0.5, hscale=1.3):
    """Single-column figure sized so on-page text lands near 8.5pt."""
    s = apply_style(scale)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(22 * s * wratio, 12 * s * hscale))
    return fig, axes, s


def on_page_font(wratio, target_w=COL_W, base=60):
    return base * target_w / (22.0 * wratio)


def axes_top_legend(ax, ncol=2, y=1.02, handles=None, labels=None, **kw):
    """Rasengan's fallback when a panel has no room: legend ABOVE the axes.

    reproduce/figure_{9,10,12,13,15,16} all use loc='upper left/center' with a
    bbox_to_anchor whose height band sits above the axes and mode='expand', so
    the key never overprints data. Font size comes from the rc ladder
    (BASE['legfs']), which is deliberately LARGER than the axis-label size.
    """
    args = (handles, labels) if handles is not None else ()
    return ax.legend(*args, loc="lower left", ncol=ncol, frameon=False,
                     bbox_to_anchor=(0.0, y, 1.0, 0.2), mode="expand",
                     borderaxespad=0.0, **kw)
