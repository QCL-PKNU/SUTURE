"""fig_e9 — objective-blind SUTURE_C on hardware (ibm_marrakesh), single column.

(a) per-seed chain-best measured conflicts, three arms paired within seed
    (36q soft-edge, regime II: all arms alive, so quality is comparable);
(b) median trajectory of best measured conflicts along the chain (same cell);
(c) chain survival vs hard-edge instance size, where the transition circuit
    uses roughly 400x as many two-qubit gates: purification is extinguished
    at both sizes while
    objective-blind SUTURE_C completes every chain. The gate count and the
    zero-feasible-measurement fact are stated in the manuscript prose, not
    annotated here (user request: no duplication).
Panels (a)/(b) share one arm key; panel (c) plots the two circuit-fed arms.
Reads only its own data/ directory.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from rasengan_style import apply_style, save_fig, check_overlaps, ygrid, PAL, BASE, COL_WRATIO

D = os.path.dirname(os.path.abspath(__file__))
fin = pd.read_csv(f"{D}/data/a4_final_b36.csv")
ch = pd.read_csv(f"{D}/data/a4_chain_b36.csv")
h48 = pd.read_csv(f"{D}/data/a3_final_hard48.csv")
h78 = pd.read_csv(f"{D}/data/a3_final_hard78.csv")
OUT = f"{D}/out"

ARMS = ["purify", "synd_c", "synd_g"]
LBL = {"purify": "purify", "synd_c": r"SUTURE$_C$",
       "synd_g": r"SUTURE$_G$",
       }
COL = {"purify": PAL["navy"], "synd_c": PAL["red"], "synd_g": PAL["green"]}
MK = {"purify": "v", "synd_c": "o", "synd_g": "s"}

# 2x2 grid: (a),(b) on the top row; (c) spans the full column width below.
s = apply_style(0.5)
fig = plt.figure(figsize=(22 * s * COL_WRATIO, 12 * s * 1.82))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.40])
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])

# ---- (a) paired per-seed chain best (unchanged) ----
ax = ax_a
piv = fin.pivot(index="seed", columns="arm", values="best_conf")
x = np.arange(len(ARMS))
for seed in piv.index:
    ax.plot(x, [piv.loc[seed, a] for a in ARMS], "-", color="#BBBBBB",
            lw=BASE["grid_lw"] * s, zorder=1, alpha=0.85)
for i, a in enumerate(ARMS):
    y = piv[a].values
    jitter = (np.arange(len(y)) - len(y) / 2) * 0.012
    ax.plot(np.full(len(y), x[i]) + jitter, y, MK[a], color=COL[a],
            ms=BASE["mks"] * s, mec="black", mew=BASE["mklw"] * s, ls="none",
            zorder=3, label=LBL[a])
    ax.plot([x[i] - 0.22, x[i] + 0.22], [np.median(y)] * 2, "-",
            color="black", lw=BASE["linelw"] * s, zorder=4)
ax.set_xticks(x); ax.set_xticklabels(["purify", r"S$_C$", r"S$_G$"])
ax.set_ylabel("chain-best conflicts")
ax.set_xlim(-0.4, len(ARMS) - 0.6)
ax.set_title("(a)", loc="left", fontsize=68 * s, fontweight="bold", pad=4)
ygrid(ax, s)
_h, _l = ax.get_legend_handles_labels()

# ---- (b) median trajectory (unchanged) ----
ax = ax_b
for a in ARMS:
    sub = ch[ch.arm == a].dropna(subset=["best_conflicts_measured"])
    med = sub.groupby("t").best_conflicts_measured.median()
    ax.plot(med.index / ch.t.max(), med.values, "-", color=COL[a],
            lw=BASE["linelw"] * s, label=LBL[a])
ax.set_xlabel("chain position (fraction)")
ax.set_ylabel("median best conflicts")
ax.set_title("(b)", loc="left", fontsize=68 * s, fontweight="bold", pad=4)
ygrid(ax, s)

# ---- (c) survival vs hard-edge instance size ----
ax = ax_c
CARMS = ["purify", "synd_c"]
cells = [("48q", h48, 14), ("78q", h78, 24)]
xc = np.arange(len(cells))
w = 0.30
for j, a in enumerate(CARMS):
    vals = [float(df[df.arm == a].survived.mean()) for _, df, _ in cells]
    off = (j - 0.5) * w
    ax.bar(xc + off, vals, w, color=COL[a], edgecolor="black",
           lw=BASE["mklw"] * s)
    for xi, v in zip(xc + off, vals):
        # a zero bar is invisible; mark the death explicitly so "0/12" is a
        # drawn result rather than blank space
        if v < 0.02:
            ax.plot([xi], [0], "x", color=COL[a], ms=BASE["mks"] * s * 0.9,
                    mew=BASE["linelw"] * s, zorder=5, clip_on=False)
        ax.text(xi, v + 0.035, f"{int(round(v * 12))}/12", ha="center",
                va="bottom", fontsize=30 * s)
# The shared key shows MARKERS while this panel draws BARS, so identify the
# arms directly under each bar instead, reusing panel (a)'s abbreviations.
import matplotlib.ticker as mticker
ax.set_xticks(xc)
ax.set_xticklabels([f"{n}\n({T} seg.)" for n, _, T in cells])
ax.tick_params(axis="x", which="major", pad=BASE["tickfs"] * s * 1.55)
minor_pos, minor_lab = [], []
for k, _ in enumerate(cells):
    for j, a in enumerate(CARMS):
        minor_pos.append(xc[k] + (j - 0.5) * w)
        minor_lab.append("purify" if a == "purify" else r"S$_C$")
ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_pos))
ax.xaxis.set_minor_formatter(mticker.FixedFormatter(minor_lab))
ax.tick_params(axis="x", which="minor", length=0, labelsize=BASE["tickfs"] * s)
ax.set_xlabel("hard-edge size")
ax.set_ylabel("completed chains")
ax.set_ylim(0, 1.30)
ax.set_yticks([0.0, 0.5, 1.0])
# autoscale leaves only 0.08 data units (4.5% of span) past the outer bar
# edges, so the bar strokes visually merge with the panel spines
ax.set_xlim(xc[0] - 0.58, xc[-1] + 0.58)
ax.set_title("(c)", loc="left", fontsize=68 * s, fontweight="bold", pad=4)
ygrid(ax, s)

fig.legend(_h, _l, frameon=False, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, 0.925), borderaxespad=0.0,
           columnspacing=1.0, handletextpad=0.35, labelspacing=0.25)
fig.tight_layout(rect=[0, 0, 1, 0.915])
fig.subplots_adjust(wspace=0.42, hspace=0.44)
check_overlaps(fig)
save_fig(fig, OUT, "fig_e9")
