"""Fig E5 (stage 1 / D1 consequence): encoding co-design — detectability
beats qubit count. (a) single-flip detect rate one-hot vs domain-wall (bars,
Rasengan bar idiom); (b) end-to-end proper rate of dw+SUTURE vs one-hot+SUTURE."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasengan_style import (figure, col_figure, ygrid, bar_kw, save_fig, check_overlaps,
                            PAL, MK, LS)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

ef = pd.read_csv(f"{D}/e5a_flip_detectability.csv")
dwc = pd.read_csv(f"{D}/e5b_dw_chain.csv")

fig, axes, s = col_figure(ncols=2, hscale=1.45)

# (a) detect rate by encoding (pooled over p; bars)
ax = axes[0]
det = ef.groupby("encoding").agg(detect=("detect_rate", "mean"),
                                 qubits=("qubits", "first")).reindex(["one-hot", "domain-wall"])
x = np.arange(2)
for i, (enc, row) in enumerate(det.iterrows()):
    kw = bar_kw(i, s, palette=[PAL["red"], PAL["navy"]])
    ax.bar(x[i], row.detect, 0.55, **kw)
    ax.text(x[i], row.detect + 0.03, f"{row.detect:.2f}", ha="center", fontsize=34 * s)
ax.set_xticks(x)
ax.set_xticklabels([f"one-hot\n({int(det.qubits.iloc[0])} qb)",
                    f"domain-wall\n({int(det.qubits.iloc[1])} qb)"])
ax.set_ylabel("per-flip detect rate\n(i.i.d. flip channel)")
ax.set_ylim(0, 1.12)
ax.set_title("(a)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ygrid(ax, s)

# (b) end-to-end consequence
ax = axes[1]
combos = [("one-hot", "rec_synd", PAL["red"], "one-hot + SUTURE$_G$"),
          ("one-hot", "purify", PAL["navy"], "one-hot + purify"),
          ("domain-wall", "rec_synd", PAL["redt"], "domain-wall + SUTURE$_G$"),
          ("domain-wall", "purify", PAL["gray"], "domain-wall + purify")]
for i, (enc, arm, col, lbl) in enumerate(combos):
    sub = dwc[(dwc.encoding == enc) & (dwc.arm == arm)].sort_values("lam")
    if not len(sub):
        continue
    ax.plot(sub.lam, sub.proper_seen, color=col, linestyle=LS[i], marker=MK[i],
            markerfacecolor=col, markeredgecolor="black", label=lbl)
ax.set_xscale("log")
ax.set_xlabel(r"noise scale $\lambda$")
ax.set_ylabel("proper-coloring measurement rate")
ax.set_ylim(-0.04, 1.06)
ax.set_title("(b)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ygrid(ax, s)
_h_b, _l_b = ax.get_legend_handles_labels()

fig.legend(_h_b, _l_b, frameon=False, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, 0.86), borderaxespad=0.0,
           columnspacing=0.8, handletextpad=0.35, labelspacing=0.2)
fig.tight_layout(rect=[0, 0, 1, 0.84])
fig.subplots_adjust(wspace=0.42)
save_fig(fig, OUT, "fig_e5")
check_overlaps(fig)
