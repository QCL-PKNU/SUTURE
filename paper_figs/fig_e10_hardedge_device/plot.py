"""SUPERSEDED — merged into fig_e9_blind_device (single-column 3-panel).
Panel (a) here became fig_e9 panel (c) with instance size on the x-axis;
panel (b) (shots inside the repair budget) was dropped from the figure —
its numbers are stated in the manuscript text. Kept for provenance only;
the manuscript no longer includes fig_e10.pdf. Do not rebuild.
fig_e10 — hard-edge objective-blind SUTURE_C on hardware (48q, 78q).

(a) chain survival vs segment at 78q, where the A0 junk control dies and the
    liveness result is therefore attributable;
(b) fraction of measured shots inside the decoder's repair budget, device
    output vs the A0 control, at both sizes.
No feasible string was measured in either cell, so this figure carries a
LIVENESS result only — there is no quality endpoint to plot.
Reads only its own data/ directory.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from rasengan_style import col_figure, save_fig, check_overlaps, ygrid, PAL, BASE

D = os.path.dirname(os.path.abspath(__file__))
c48 = pd.read_csv(f"{D}/data/a3_chain_hard48.csv")
c78 = pd.read_csv(f"{D}/data/a3_chain_hard78.csv")
OUT = f"{D}/out"

LBL = {"purify": "Rasengan (purify)", "synd_c": r"SUTURE$_C$",
       "junk": "A0 control (decoder on noise)"}
COL = {"purify": PAL["navy"], "synd_c": PAL["red"], "junk": PAL["gray"]}

fig, axes, s = col_figure(ncols=2, hscale=1.30)

# (a) survival at 78q
ax = axes[0]
T = int(c78.t.max()) + 1
# purify and the A0 control both sit at zero from segment 0; offset them
# vertically by a hair so both are visible, and mark the death point.
OFF = {"purify": 0.022, "synd_c": 0.0, "junk": -0.022}
DASH = {"purify": "-", "synd_c": "-", "junk": (0, (4, 2))}
for a in ("purify", "synd_c", "junk"):
    sub = c78[c78.arm == a]
    surv = np.array([sub[sub.t == t].alive.sum() / 12 for t in range(T)], dtype=float)
    ax.plot(range(T), surv + OFF[a], ls=DASH[a], color=COL[a],
            lw=BASE["linelw"] * s, label=LBL[a])
    if surv[0] == 0:
        ax.plot([0], [OFF[a]], "o", color=COL[a], ms=BASE["mks"] * s,
                mec="black", mew=BASE["mklw"] * s, zorder=5)
ax.set_xlabel("segment")
ax.set_ylabel("chains alive (fraction of 12)")
ax.set_ylim(-0.06, 1.06)
ax.set_title("(a)", loc="left", fontsize=68 * s, fontweight="bold", pad=4)
ygrid(ax, s)
_h, _l = ax.get_legend_handles_labels()

# (b) fraction of shots within the repair budget, device vs A0
ax = axes[1]
cells = [("48q", c48), ("78q", c78)]
x = np.arange(len(cells))
dev = [cc[(cc.arm == "synd_c") & (cc.t == 0)].frac_in_budget.mean() for _, cc in cells]
jnk = [cc[(cc.arm == "junk") & (cc.t == 0)].frac_in_budget.mean() for _, cc in cells]
ax.bar(x - 0.19, dev, 0.34, color=PAL["red"], edgecolor="black", lw=BASE["mklw"] * s)
ax.bar(x + 0.19, jnk, 0.34, color=PAL["gray"], edgecolor="black",
       lw=BASE["mklw"] * s, hatch="///")
for xi, val in zip(x + 0.19, jnk):
    if val < 0.01:
        ax.text(xi, 0.012, "0", ha="center", va="bottom", fontsize=30 * s)
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in cells])
ax.set_ylabel("shots within repair budget")
ax.set_xlabel("hard-edge instance size")
ax.set_title("(b)", loc="left", fontsize=68 * s, fontweight="bold", pad=4)
ygrid(ax, s)

fig.legend(_h, _l, frameon=False, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, 0.905), borderaxespad=0.0,
           columnspacing=0.9, handletextpad=0.35, labelspacing=0.22)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.subplots_adjust(wspace=0.42)
check_overlaps(fig)
save_fig(fig, OUT, "fig_e10")
