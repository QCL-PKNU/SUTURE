"""Fig E7 (hardware, ibm_marrakesh): (a) A1 block-locality bars (24/36q);
(b) A2 quality trajectories 36/48q + 72q survival inset story as third
panel; (c) Fig-12-style stacked latency breakdown (Rasengan bar idiom:
classical solid, quantum hatched)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasengan_style import (figure, ygrid, bar_kw, save_fig, check_overlaps,
                            PAL, LS, MK)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

bl = pd.read_csv(f"{D}/e7a_blocklocality.csv")
chains = pd.read_csv(f"{D}/e7b_chains.csv")
lat = pd.read_csv(f"{D}/e7d_latency.csv")

fig, axes, s = figure(scale=0.55, ncols=3, wratio=2.6)

# (a) block-locality
ax = axes[0]
x = np.arange(len(bl))
ax.bar(x - 0.19, bl.blk1, 0.34, label="1 block",
       **bar_kw(0, s, palette=[PAL["red"]]))
ax.bar(x + 0.19, bl.blk2, 0.34, label=r"$\leq$2 blocks",
       **bar_kw(1, s, palette=[PAL["gray"], PAL["redt"]]))
_h_a, _l_a = ax.get_legend_handles_labels()
for xi, (b1, b2) in enumerate(zip(bl.blk1, bl.blk2)):
    ax.text(xi - 0.19, b1 + 0.025, f"{b1:.2f}", ha="center", fontsize=30 * s)
    ax.text(xi + 0.19, b2 + 0.025, f"{b2:.2f}", ha="center", fontsize=30 * s)
ax.axhline(0.6, color="black", linestyle="--", linewidth=3 * s)
ax.text(0.5, 0.53, "pre-specified gate (0.6)", fontsize=27 * s, ha="center")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(n)}q" for n in bl.n_qubits])
ax.set_ylabel("fraction of infeasible strings")
ax.set_ylim(0, 1.14)
ax.set_title("(a)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ygrid(ax, s)

# (b) A2 quality trajectories at 36/48/72q (SUTURE solid red, purify navy)
ax = axes[1]
for j, nq in enumerate((36, 48, 72)):
    for arm, col in (("purify", PAL["navy"]), ("synd", PAL["red"])):
        sub = chains[(chains.n_qubits == nq) & (chains.arm == arm)].dropna(subset=["best"])
        # survivor-bias guard: draw the quality curve only while >=6 of 12
        # chains survive -- past that the median reflects a lone survivor
        sub = sub[sub.alive >= 0.5]
        if not len(sub):
            continue
        xfrac = sub.t / sub.segments.iloc[0]
        ax.plot(xfrac, sub.best, color=col, linestyle=LS[j], linewidth=(2.4 + 0.8 * j) * s * 2,
                alpha=0.55 + 0.15 * j,
                label=f"{nq}q {'purify' if arm == 'purify' else 'SUTURE$_G$'}")
ax.set_xlabel("chain position (fraction of segments)")
ax.set_ylabel("best measured conflicts (median)")
ax.set_title("(b)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ygrid(ax, s)
_h_b, _l_b = ax.get_legend_handles_labels()

# (c) latency breakdown, Rasengan fig-12 idiom: classical solid + quantum hatched
ax = axes[2]
comps_cl = [("decode_s", "decode/repair"), ("register_s", "register update"),
            ("transpile_s", "transpile"), ("build_s", "circuit build")]
comps_q = [("quantum_s", "QPU execution"), ("queue_s", "queue")]
cols_cl = [PAL["red"], PAL["cream"], PAL["green"], PAL["navy"]]
cols_q = [PAL["gray"], PAL["greent"]]
x = np.arange(len(lat))
bot = np.zeros(len(lat))
handles = []
for (key, lbl), col in zip(comps_cl, cols_cl):
    ax.bar(x, lat[key], 0.5, bottom=bot, color=col, edgecolor="black", linewidth=2.4 * s)
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="black", label=f"c: {lbl}"))
    bot += lat[key].values
for (key, lbl), col in zip(comps_q, cols_q):
    ax.bar(x, lat[key], 0.5, bottom=bot, color=col, edgecolor="black",
           linewidth=2.4 * s, hatch="/")
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="black",
                                 hatch="/", label=f"q: {lbl}"))
    bot += lat[key].values
for xi, tot in enumerate(bot):
    ax.text(xi, tot + 3, f"{tot:.0f}s", ha="center", fontsize=32 * s)
ax.set_xticks(x)
ax.set_xticklabels(["purify", "SUTURE$_G$"])
ax.set_ylabel("cumulative runtime, 48 rounds (s)")
ax.set_ylim(0, 175)
ax.set_title("(c)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ygrid(ax, s)
_h_c, _l_c = handles, [h.get_label() for h in handles]

fig.legend(_h_a, _l_a, frameon=False, loc="upper left", ncol=1,
           bbox_to_anchor=(0.02, 1.06), borderaxespad=0.0,
           handletextpad=0.35, labelspacing=0.18)
fig.legend(_h_b, _l_b, frameon=False, loc="upper left", ncol=3,
           bbox_to_anchor=(0.27, 1.06), borderaxespad=0.0,
           columnspacing=0.7, handletextpad=0.35, labelspacing=0.18)
fig.legend(_h_c, _l_c, frameon=False, loc="upper left", ncol=2,
           bbox_to_anchor=(0.735, 1.06), borderaxespad=0.0,
           columnspacing=0.8, handletextpad=0.35, labelspacing=0.18)
fig.tight_layout(rect=[0, 0, 1, 0.78])
save_fig(fig, OUT, "fig_e7")
check_overlaps(fig)
