"""Fig E6 (generalization, 3 panels): (a) stage-1 validation — measured
single-flip recovery vs E[1/R], 70 instances / 4 families; (b) stage-3
premise — purification cliff atlas across families; (c) stage-3 verdict —
peak death-window discordance, 0-loss at every scale."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasengan_style import figure, ygrid, save_fig, check_overlaps, PAL, MK

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

prof = pd.read_csv(f"{D}/e6a_profiler.csv")
cliff = pd.read_csv(f"{D}/e6b_cliffs.csv")
disc = pd.read_csv(f"{D}/e6c_discordance.csv")

FAM = {"gcp": (PAL["red"], "o", "GCP"), "flp": (PAL["navy"], "s", "FLP"),
       "scp": (PAL["green"], "^", "SCP"), "kpp": ("#7B5BA6", "D", "KPP")}

def fam_of(x):
    for k in FAM:
        if str(x).startswith(k):
            return k
    return "gcp"

fig, axes, s = figure(scale=0.55, ncols=3, wratio=2.6)

# (a) profiler prediction
ax = axes[0]
for k, (col, mk, lbl) in FAM.items():
    sub = prof[prof.family.str.startswith(k)]
    ax.scatter(sub.inv_R_mean, sub.rec1_emp, s=210 * s, color=col, marker=mk,
               edgecolor="black", linewidth=2 * s, zorder=3, label=lbl)
ax.plot([0.45, 1.03], [0.45, 1.03], color="#777", linestyle=":", linewidth=3 * s)
sil = prof[prof.silent_frac > 0]
if len(sil):
    ax.annotate("zero column in $C$\n(static check rejects)",
                xy=(float(sil.inv_R_mean.iloc[0]), float(sil.rec1_emp.iloc[0])),
                xytext=(0.58, 0.86), fontsize=27 * s, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color="#555", lw=2 * s))
ax.set_xlabel(r"predicted recovery $\mathbb{E}[1/R]$")
ax.set_ylabel("measured recovery", labelpad=8)
ax.set_title("(a)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ygrid(ax, s)


# (b) cliff atlas
ax = axes[1]
for k, (col, mk, lbl) in FAM.items():
    sub = cliff[cliff.scale.apply(fam_of) == k].sort_values("n")
    if not len(sub):
        continue
    y = sub.purify_cliff.fillna(70)   # no-death cells drawn at top band
    ax.plot(sub.n, np.clip(y, None, 70), color=col, marker=mk, linestyle="--",
            markerfacecolor=col, markeredgecolor="black", label=lbl)
rd = cliff[cliff.repair_cliff.notna()]
if len(rd):
    ax.scatter(rd.n, rd.repair_cliff, s=340 * s, marker="X", color="black",
               zorder=4, label="repair-arm death\n(weight budget)")
ax.axhspan(1, 4, color="#EEE0D0", alpha=0.7, zorder=0)
ax.text(16, 1.55, "realistic device noise band", fontsize=26 * s, color="#8a6a4a")
ax.set_yscale("log")
ax.set_yticks([1, 2, 4, 8, 16, 32, 64])
ax.set_yticklabels(["1", "2", "4", "8", "16", "32", ">grid"])
ax.set_xlabel("problem size (qubits)")
ax.set_ylabel(r"purification death $\lambda$ (survival < 0.5)")
ax.set_title("(b)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ygrid(ax, s)
_h_all, _l_all = ax.get_legend_handles_labels()

# (c) peak discordance WITHIN each scale's death window (lam >= its
# purification cliff); scales whose purification never dies on the grid
# have no death window and are excluded from this panel.
ax = axes[2]
dw = disc.merge(cliff[["scale", "purify_cliff"]], on="scale", how="left")
dw = dw[dw.purify_cliff.notna() & (dw.lam >= dw.purify_cliff)]
peaks = dw.loc[dw.groupby("scale").apply(
    lambda g_: (g_.win - g_.lose).idxmax(), include_groups=False)]
assert (peaks.lose == 0).all(), "zero-loss claim violated in death window"
rng = np.random.default_rng(5)
for k, (col, mk, lbl) in FAM.items():
    sub = peaks[peaks.family.str.startswith(k)]
    if not len(sub):
        continue
    ax.scatter(sub.n + rng.uniform(-0.6, 0.6, len(sub)), sub.win, s=230 * s,
               color=col, marker=mk, edgecolor="black", linewidth=2 * s,
               zorder=3, label=lbl)
    ax.scatter(sub.n, -sub.lose, s=230 * s, facecolor="white", edgecolor=col,
               marker=mk, linewidth=3 * s, zorder=3)
ax.axhline(0, color="black", linewidth=2 * s)
ax.text(30, -4.5, "losses (all zero at every peak)", fontsize=28 * s)
ax.set_xlabel("problem size (qubits)")
ax.set_ylabel("peak death-window discordant seeds\n(wins above axis, losses below)")
n_dw = peaks.scale.nunique()
ax.set_title("(c)", loc="left", fontsize=68 * s, fontweight="bold", pad=6)
ax.set_ylim(-8, None)
ygrid(ax, s)


_l_all = [l.replace("\n", " ") for l in _l_all]
fig.legend(_h_all, _l_all, frameon=False, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, 0.97), borderaxespad=0.0,
           columnspacing=1.2, handletextpad=0.4)
fig.tight_layout(rect=[0, 0, 1, 0.88])
save_fig(fig, OUT, "fig_e6")
check_overlaps(fig)
