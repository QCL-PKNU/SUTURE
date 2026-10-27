"""Fig E1 (stage 2 core): go/no-go separation — proper-coloring measurement
rate by arm across noise, gcp_v8 (24q) S=64. Rasengan line idiom."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasengan_style import (figure, ygrid, line_kw, save_fig, check_overlaps,
                            ARM_COLOR, ARM_LABEL, top_legend, arm_label)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

df = pd.read_csv(f"{D}/e1_proper_by_arm_lam.csv")
ARMS = ["rec_synd", "purify", "purify_restart", "purify_last",
        "rec_nearest", "rec_uniform", "junk_synd"]

fig, ax, s = figure(scale=0.5)
for i, arm in enumerate(ARMS):
    sub = df[df.arm == arm].sort_values("lam")
    ax.plot(sub.lam, sub.proper_seen, label=arm_label(arm, "G"),
            **line_kw(i, color=ARM_COLOR[arm]))
ax.set_xscale("log")
ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
ax.set_xlabel(r"noise scale $\lambda$")
ax.set_ylabel("proper-coloring measurement rate")
ax.set_ylim(-0.04, 1.06)
ygrid(ax, s)
top_legend(fig, ncol=2, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, OUT, "fig_e1")
check_overlaps(fig)
