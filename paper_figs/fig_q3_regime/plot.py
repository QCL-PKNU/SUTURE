"""fig_q3 (merged, 11-page restructure): the deployment story in one figure.
(a) repair loses at low noise (e3a); (b) the crossover map over (size, lambda)
(e3b, grid-Holm); (c) scale to 120q at fixed noise (e2a).
Reads only its own data/ directory (copies of e3/e2 panel data)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from rasengan_style import (figure, ygrid, save_fig, check_overlaps, PAL,
                            ARM_COLOR, MK, LS, arm_label)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

low = pd.read_csv(f"{D}/e3b_lownoise.csv")
kv = pd.read_csv(f"{D}/e3a_kv_dmed.csv")
g = pd.read_csv(f"{D}/e2a_scale_grid.csv")

fig, axes, s = figure(scale=0.55, ncols=3, wratio=2.6)

# ---- (a) low-noise margins by family
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
ax.text(0.45, -1.08, "shaded: purify better\n(30/50 cells)", fontsize=27 * s, ha="left")
ax.set_xticks(range(len(fams))); ax.set_xticklabels(["GCP", "FLP", "SCP"])
ax.set_ylim(-1.35, 1.25)
ax.set_ylabel(r"$\Delta$ARG = purify $-$ repair", labelpad=8)
ax.set_xlabel(r"low-noise cells ($\lambda \leq 4$)")
ax.set_title("(a)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ygrid(ax, s)

# ---- (b) crossover map
ax = axes[1]
nsq = sorted(kv.n_qubits.dropna().unique())
lams = sorted(kv.lam.unique())
def holm(pv):
    pv = np.asarray(pv, float); order = np.argsort(pv); m = len(pv)
    adj = np.empty(m); run = 0.0
    for k, idx in enumerate(order):
        run = max(run, (m - k) * pv[idx]); adj[idx] = min(run, 1.0)
    return adj
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
ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:g}" for l in lams])
ax.set_yticks(range(len(nsq))); ax.set_yticklabels([f"{int(n)}q" for n in nsq])
ax.set_xlabel(r"noise scale $\lambda$")
ax.set_ylabel("problem size")
ax.set_title("(b)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
_h_b = [
    mlines.Line2D([], [], marker="s", color=PAL["red"], markeredgecolor=PAL["red"],
                  linestyle="", markersize=16 * s * 1.4, label="repair better (grid-Holm $p<$.05)"),
    mlines.Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=PAL["red"],
                  color="white", linestyle="", markersize=13 * s * 1.4, label="repair better (null)"),
    mlines.Line2D([], [], marker="o", color=PAL["navy"], markeredgecolor=PAL["navy"],
                  linestyle="", markersize=16 * s * 1.4, label="purify better (grid-Holm $p<$.05)"),
    mlines.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=PAL["navy"],
                  color="white", linestyle="", markersize=13 * s * 1.4, label="purify better (null)"),
]

# ---- (c) scale to 120q
ax = axes[2]
for j, lam in enumerate((4.0, 8.0)):
    for i, arm in enumerate(("purify", "rec_synd")):
        sub = g[(g.arm == arm) & (g.lam == lam)].groupby("n").proper_seen.mean()
        ax.plot(sub.index, sub.values,
                color=ARM_COLOR[arm], linestyle=LS[j], marker=MK[j],
                markerfacecolor=ARM_COLOR[arm], markeredgecolor="black",
                label=f"{arm_label(arm, 'G').split(' (')[0]}, $\\lambda$={lam:g}")
ax.set_xlabel("problem size (qubits)")
ax.set_ylabel("proper-coloring rate")
ax.set_ylim(-0.04, 1.06)
ax.set_title("(c)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ygrid(ax, s)
_h_c, _l_c = ax.get_legend_handles_labels()

# two figure-level keys on one top band
fig.legend(handles=_h_b, frameon=False, loc="upper left", ncol=2,
           bbox_to_anchor=(0.055, 1.045), borderaxespad=0.0,
           labelspacing=0.22, handletextpad=0.35, columnspacing=0.8)
fig.legend(_h_c, _l_c, frameon=False, loc="upper right", ncol=2,
           bbox_to_anchor=(0.985, 1.045), borderaxespad=0.0,
           labelspacing=0.22, handletextpad=0.35, columnspacing=0.8)
fig.tight_layout(rect=[0, 0, 1, 0.80])
check_overlaps(fig)
save_fig(fig, OUT, "fig_q3")
