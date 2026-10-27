"""Fig E4 (stages 2+3): (a) hard-edge variant — best measured cost by arm
across noise, hard_v8 (54q incl. slack); (b) noise-assisted optimum — SUTURE
proper-rate vs lambda across training budgets (mechanism panel)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasengan_style import (figure, col_figure, col_figure, ygrid, line_kw, save_fig, check_overlaps,
                            PAL, ARM_COLOR, ARM_LABEL, MK, LS, arm_label)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

he = pd.read_csv(f"{D}/e4a_hardedge.csv")
tr = pd.read_csv(f"{D}/e4b_training.csv")

fig, axes, s = col_figure(ncols=2, hscale=1.30)

# (a) hard-edge best measured cost (finite cells only; E* line)
ax = axes[0]
ARMS = ["rec_synd", "purify", "purify_restart", "rec_nearest"]
for i, arm in enumerate(ARMS):
    sub = he[(he.arm == arm) & (he.n_finite >= 6)].sort_values("lam")
    if not len(sub):
        continue
    ax.plot(sub.lam, sub.best_cost_med, label=arm_label(arm, "C"),
            **line_kw(i, color=ARM_COLOR[arm]))
ax.axhline(25, color=PAL["orange"], linestyle="--", linewidth=3 * s)
ax.text(4.4, 25.7, r"optimum $E^*$=25", fontsize=30 * s, color=PAL["orange"])
ax.set_xscale("log")
import matplotlib.ticker as mticker
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xticks([1, 2, 4, 8])
ax.set_xlabel(r"noise scale $\lambda$")
ax.set_ylabel("best measured cost (median)")
ax.set_title("(a)", loc="left", x=0.02, fontsize=68 * s, fontweight="bold", pad=4)
ygrid(ax, s)
_h_a, _l_a = ax.get_legend_handles_labels()

# (b) noise-assisted optimum across training budgets
ax = axes[1]
budgets = sorted(tr.budget.unique())
for i, b in enumerate(budgets):
    sub = tr[tr.budget == b].sort_values("lam")
    shade = 0.3 + 0.7 * i / max(len(budgets) - 1, 1)
    ax.plot(np.clip(sub.lam, 0.25, None), sub.proper_seen, color=PAL["red"],
            alpha=shade, linestyle=LS[i % len(LS)], marker=MK[i % len(MK)],
            markerfacecolor=PAL["red"], markeredgecolor="black",
            label=f"budget {b}")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xticks([0.5, 1, 2, 4, 8, 16])
ax.set_xlabel(r"noise scale $\lambda$")
ax.set_ylabel("SUTURE$_G$ proper-coloring rate")
ax.set_title("(b)", loc="left", x=0.02, fontsize=68 * s, fontweight="bold", pad=4)
ax.set_ylim(-0.04, 1.06)
ygrid(ax, s)
_h_b, _l_b = ax.get_legend_handles_labels()

fig.legend(_h_a, _l_a, frameon=False, loc="upper left", ncol=1,
           bbox_to_anchor=(0.005, 1.04), borderaxespad=0.0,
           handletextpad=0.4, labelspacing=0.18)
fig.legend(_h_b, _l_b, frameon=False, loc="upper right", ncol=2,
           bbox_to_anchor=(0.998, 1.04), borderaxespad=0.0,
           columnspacing=0.6, handletextpad=0.25, labelspacing=0.18)
fig.tight_layout(rect=[0, 0, 1, 0.76])
fig.subplots_adjust(wspace=0.42)
save_fig(fig, OUT, "fig_e4")
check_overlaps(fig)
