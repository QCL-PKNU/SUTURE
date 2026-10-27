"""Fig E3 (stage 3): the honest two-sided quality story. (a) low-noise cells:
purify-vs-repair paired ARG margins across families (the cost of repair
where purification is healthy); (b) KV flip map: which (size, lambda) cells
favor which arm (sign of d_med, significance-marked)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasengan_style import figure, col_figure, ygrid, save_fig, check_overlaps, PAL

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

low = pd.read_csv(f"{D}/e3b_lownoise.csv")
kv = pd.read_csv(f"{D}/e3a_kv_dmed.csv")

fig, axes, s = col_figure(ncols=2)

# (a) low-noise margins by family (dot strip, purify-better shaded)
ax = axes[0]
fams = ["gcp", "flp", "scp"]
FAMC = {"gcp": PAL["red"], "flp": PAL["navy"], "scp": PAL["green"]}
rng = np.random.default_rng(3)
for i, fam in enumerate(fams):
    sub = low[low.family.str.startswith(fam)]
    x = np.full(len(sub), i, dtype=float) + rng.uniform(-0.14, 0.14, len(sub))
    sig = sub.p < 0.05
    ax.scatter(x[sig.values], sub.d_med[sig], s=170 * s, color=FAMC[fam],
               edgecolor="black", linewidth=2 * s, zorder=3, marker="o")
    ax.scatter(x[~sig.values], sub.d_med[~sig], s=170 * s, facecolor="white",
               edgecolor=FAMC[fam], linewidth=3 * s, zorder=3, marker="o")
ax.axhline(0, color="black", linestyle=":", linewidth=2 * s)
ax.axhspan(-1.35, 0, color="#DDDDDD", alpha=0.5, zorder=0)
ax.text(0.55, -1.05, "shaded: purify better (30/50 cells)", fontsize=30 * s, ha="left")
ax.set_xticks(range(len(fams)))
ax.set_xticklabels(["GCP", "FLP", "SCP"])
ax.set_ylim(-1.35, 1.25)
ax.set_ylabel(r"$\Delta$ARG = purify $-$ repair", labelpad=8)
ax.set_xlabel(r"low-noise cells ($\lambda \leq 4$)")
ax.set_title("(a)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ygrid(ax, s)

# (b) KV flip map (size x lambda cells, sign + significance)
ax = axes[1]
nsq = sorted(kv.n_qubits.dropna().unique())
lams = sorted(kv.lam.unique())
def holm(pv):
    pv = np.asarray(pv, float); order = np.argsort(pv); m = len(pv)
    adj = np.empty(m); run = 0.0
    for k, idx in enumerate(order):
        run = max(run, (m - k) * pv[idx]); adj[idx] = min(run, 1.0)
    return adj

# Holm correction ACROSS the lambda grid per scale: this is the criterion
# that defines lambda_q, so the map must use it (within-cell Holm already
# applied upstream in the paired tests).
kv = kv.copy(); kv["p_grid"] = np.nan
for nq, grp in kv.groupby("n_qubits"):
    ok = grp.p.notna() & grp.d_med.notna()
    kv.loc[grp.index[ok], "p_grid"] = holm(grp.loc[ok, "p"].values)

for _, r_ in kv.iterrows():
    if not np.isfinite(r_.d_med):
        continue
    xi, yi = lams.index(r_.lam), nsq.index(r_.n_qubits)
    win = r_.d_med > 0
    sig = r_.p_grid < 0.05 if np.isfinite(r_.p_grid) else False
    col = PAL["red"] if win else PAL["navy"]
    ax.scatter(xi, yi, s=(560 if sig else 220) * s, marker="s" if win else "o",
               color=col if sig else "white", edgecolor=col, linewidth=3 * s, zorder=3)
ax.set_xticks(range(len(lams)))
ax.set_xticklabels([f"{l:g}" for l in lams])
ax.set_yticks(range(len(nsq)))
ax.set_yticklabels([f"{int(n)}q" for n in nsq])
ax.set_xlabel(r"noise scale $\lambda$")
ax.set_ylabel("problem size")
import matplotlib.lines as mlines
handles = [
    mlines.Line2D([], [], marker="s", color=PAL["red"], markeredgecolor=PAL["red"],
                  linestyle="", markersize=16 * s * 1.4, label="repair better (grid-Holm $p<$.05)"),
    mlines.Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=PAL["red"],
                  color="white", linestyle="", markersize=13 * s * 1.4, label="repair better (null)"),
    mlines.Line2D([], [], marker="o", color=PAL["navy"], markeredgecolor=PAL["navy"],
                  linestyle="", markersize=16 * s * 1.4, label="purify better (grid-Holm $p<$.05)"),
    mlines.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=PAL["navy"],
                  color="white", linestyle="", markersize=13 * s * 1.4, label="purify better (null)"),
]
fig.legend(handles=handles, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, 0.175), ncol=2,
          labelspacing=0.25, handletextpad=0.4, columnspacing=1.0)
ax.set_title("(b)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)

fig.tight_layout(rect=[0, 0.135, 1, 1])
fig.subplots_adjust(wspace=0.42)
save_fig(fig, OUT, "fig_e3")
check_overlaps(fig)
