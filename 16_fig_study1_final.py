"""
Final Study 1 figure using the 11-corpus dataset and cluster-robust p-values.
No title baked into image (Tao's request).
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
from matplotlib import rcParams
rcParams["font.family"] = ["Noto Sans CJK JP","DejaVu Sans"]
rcParams["font.sans-serif"] = ["Noto Sans CJK JP","DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42

chi = pd.read_csv("study1_chi2_all.csv")
cr  = pd.read_csv("study1_cluster_robust.csv")

LABELS = {"quantifier":"some/all\n一些/所有",
          "whNP":"what/which\n什么/哪",
          "anchor":"kind of\n种/类",
          "another":"another/other\n另/别的",
          "labelling":"this/that + BE\n这/那 + 是",
          "definite":"this/that + N\n这/那 + N"}
ORDER_CRIT = ["quantifier","whNP","anchor","another"]
ORDER_CTRL = ["labelling","definite"]

# We use cluster-robust p-values (more conservative/appropriate) for stars
fig, (axc, axk) = plt.subplots(1, 2, figsize=(10.8, 4.6),
                               gridspec_kw={"width_ratios":[4,2]}, sharey=True)

def plot_panel(ax, order, show_legend=False):
    x = np.arange(len(order)); w = 0.38
    sp = [chi.loc[chi.context==c,"s_p"].iloc[0] for c in order]
    bp = [chi.loc[chi.context==c,"b_p"].iloc[0] for c in order]
    ax.bar(x - w/2, sp, w, label="Superordinate", color="#4C72B0", edgecolor="k")
    ax.bar(x + w/2, bp, w, label="Basic",         color="#DD8452", edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in order], fontsize=9)
    ax.grid(True, axis="y", alpha=.3); ax.set_axisbelow(True)
    ax.set_ylim(0, 0.25)
    ax.set_yticks(np.arange(0, 0.26, 0.05))
    ax.set_yticklabels([f"{int(y*100)}%" for y in np.arange(0, 0.26, 0.05)])
    for i, c in enumerate(order):
        p_cr = cr.loc[cr.context==c, "p"].iloc[0]
        stars = "***" if p_cr < .001 else "**" if p_cr < .01 else "*" if p_cr < .05 else ""
        if stars:
            ymax = max(sp[i], bp[i])
            ax.text(i, ymax + 0.012, stars, ha="center", fontsize=12, fontweight="bold")
    if show_legend: ax.legend(loc="upper right", fontsize=10, frameon=True)

axc.set_ylabel("Proportion of target-utterances\nhosting each context", fontsize=10)
plot_panel(axc, ORDER_CRIT, show_legend=True)
plot_panel(axk, ORDER_CTRL, show_legend=False)
axc.text(0.02, 0.98, "Critical contexts",
         transform=axc.transAxes, va="top", ha="left", fontsize=11, fontweight="bold")
axk.text(0.04, 0.98, "Control contexts",
         transform=axk.transAxes, va="top", ha="left", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/analysis/fig_study1.png", dpi=200, bbox_inches="tight")
print("Saved fig_study1.png")
