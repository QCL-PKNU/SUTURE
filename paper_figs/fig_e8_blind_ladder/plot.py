"""fig_e8 — objective-blind hard-edge ladder (report 22, manuscript sec:eval-blind).

(a) proper-coloring rate vs size, pooled over lambda, with the A0-contaminated
    sizes shaded and excluded from claims.
(b) the two attribution-clean sizes by noise scale.
Reads only its own data/ directory.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rasengan_style as RS

hl = pd.read_csv(os.path.join(HERE, "data", "hard_ladder.csv"))
PAL = RS.PAL
COL = {"synd": PAL["red"], "nearest": PAL["green"], "purify_restart": PAL["gray"],
       "purify_last": "#5a6b8f", "purify": PAL["navy"], "junk_synd": "#333333"}
ORDER = ["synd", "nearest", "purify_restart", "purify_last", "purify", "junk_synd"]
LBL = {"synd": "SUTURE$_C$ (ours, blind)", "nearest": "coset-uniform (ablation)",
       "purify_restart": "purify+restart", "purify_last": "purify+last",
       "purify": "Rasengan (purify)", "junk_synd": "SUTURE-on-junk (A0)"}

fig, axes, s = RS.col_figure(ncols=2)

ax = axes[0]
for i, arm in enumerate(ORDER):
    m = hl[hl.arm == arm].groupby("n").proper_seen.mean()
    ax.plot(m.index, m.values, label=LBL[arm], **RS.line_kw(i, color=COL[arm]))
ax.axvspan(50, 72, color="#999999", alpha=0.13, zorder=0)
ax.text(60, 0.13, "A0 nonzero:\nnot attributable", fontsize=24 * s, ha="center", color="#555")
ax.set_xlabel("decision + slack qubits $n$")
ax.set_ylabel("proper-coloring measurement rate", labelpad=6)
ax.set_xticks([54, 66, 78, 90]); ax.set_ylim(-0.05, 1.14)
ax.set_title("(a)", loc="left", x=0.02, fontsize=68 * s, fontweight="bold", pad=4)
RS.ygrid(ax, s)

ax = axes[1]
sub = hl[hl.n >= 78]
for i, arm in enumerate(ORDER):
    m = sub[sub.arm == arm].groupby("lam").proper_seen.mean()
    x = np.clip(m.index.values.astype(float), 0.25, None)
    ax.plot(x, m.values, **RS.line_kw(i, color=COL[arm]))
ax.set_xscale("log")
ax.set_xlabel("noise scale $\\lambda$")
ax.set_ylabel("proper-coloring measurement rate", labelpad=6)
ax.set_ylim(-0.05, 1.14)
ax.set_title("(b)", loc="left", x=0.02, fontsize=68 * s, fontweight="bold", pad=4)
RS.ygrid(ax, s)

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, 1.04), ncol=2, columnspacing=1.2, handletextpad=0.4)
fig.tight_layout(w_pad=5.5, rect=[0, 0, 1, 0.80])
fig.subplots_adjust(wspace=0.42)
RS.save_fig(fig, os.path.join(HERE, "out"), "fig_e8")
RS.check_overlaps(fig)
