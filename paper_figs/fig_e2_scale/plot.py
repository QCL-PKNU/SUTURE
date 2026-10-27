"""Fig E2 (stage 3): scale story. (a) proper-seen vs size at fixed noise
bands, purify vs SUTURE (scale grid 36-120q); (b) completed KV grid paired
ARG margin vs lambda by size (15-55q)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasengan_style import (figure, col_figure, axes_top_legend, ygrid, line_kw, save_fig, check_overlaps,
                            PAL, ARM_COLOR, ARM_LABEL, MK, LS, arm_label)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

g = pd.read_csv(f"{D}/e2a_scale_grid.csv")
kv = pd.read_csv(f"{D}/e2b_kv_paired.csv")

fig, axes, s = col_figure(ncols=2, hscale=1.42)

# (a) proper-seen vs n at two noise bands
ax = axes[0]
for j, lam in enumerate((4.0, 8.0)):
    for i, arm in enumerate(("purify", "rec_synd")):
        sub = g[(g.arm == arm) & (g.lam == lam)].groupby("n").proper_seen.mean()
        ax.plot(sub.index, sub.values,
                color=ARM_COLOR[arm], linestyle=LS[j], marker=MK[j],
                markerfacecolor=ARM_COLOR[arm], markeredgecolor="black",
                label=f"{arm_label(arm, 'G').split(' (')[0]}, $\\lambda$={lam:g}")
ax.set_title("(a)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ax.set_xlabel("problem size (qubits)")
ax.set_ylabel("proper-coloring measurement rate")
ax.set_ylim(-0.04, 1.06)
ygrid(ax, s)
_h_a, _l_a = ax.get_legend_handles_labels()

# (b) KV paired ARG margin vs lambda per size
ax = axes[1]
order = ["kv_v3e2", "kv_v4e3", "kv_v4e4", "kv_v4e5", "kv_v5e4", "kv_v5e6"]
for i, scale_name in enumerate(order):
    sub = kv[(kv.scale == scale_name) & kv.d_med.notna() & (kv.n_pairs >= 6)].sort_values("lam")
    if not len(sub):
        continue
    shade = 0.25 + 0.75 * i / (len(order) - 1)
    ax.plot(np.clip(sub.lam, 0.25, None), sub.d_med,
            color=PAL["red"], alpha=shade, linestyle=LS[i % len(LS)],
            marker=MK[i % len(MK)], markerfacecolor=PAL["red"],
            markeredgecolor="black", label=f"{sub.n_qubits.iloc[0]}q")
ax.axhline(0, color="black", linestyle=":", linewidth=2 * s)
ax.text(0.98, 0.97, "repair better $\\uparrow$", transform=ax.transAxes,
        ha="right", va="top", fontsize=28 * s)
ax.text(0.98, 0.03, "purify better $\\downarrow$", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=28 * s)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 4, 8, 16])
ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16"])
ax.set_xlabel(r"noise scale $\lambda$")
ax.set_title("(b)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ax.set_ylabel(r"$\Delta$ARG = purify $-$ repair (median)")
ygrid(ax, s)
_h_b, _l_b = ax.get_legend_handles_labels()

# Two keys, one top band (Rasengan's fallback when panels have no room):
# arms across the left half, sizes across the right half.
fig.legend(_h_a, _l_a, frameon=False, loc="upper left", ncol=1,
           bbox_to_anchor=(0.005, 1.005), borderaxespad=0.0,
           handletextpad=0.4, labelspacing=0.18)
fig.legend(_h_b, _l_b, frameon=False, loc="upper right", ncol=2,
           bbox_to_anchor=(0.998, 1.005), borderaxespad=0.0,
           columnspacing=0.6, handletextpad=0.25, labelspacing=0.18)
fig.tight_layout(rect=[0, 0, 1, 0.79])
fig.subplots_adjust(wspace=0.42)
save_fig(fig, OUT, "fig_e2")
check_overlaps(fig)
