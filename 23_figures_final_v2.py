"""Final figures for revised manuscript (Xu + Kuperman only)."""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
from matplotlib import rcParams
rcParams["font.family"] = ["Noto Sans CJK JP","DejaVu Sans"]
rcParams["font.sans-serif"] = ["Noto Sans CJK JP","DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42

exp = pd.read_csv("/home/claude/analysis/expanded_contexts.csv")
bi  = pd.read_csv("/home/claude/analysis/bilingual_xu_kup.csv")
chi_chi2 = pd.read_csv("/home/claude/analysis/study1_chi2_all.csv")
chi_cr   = pd.read_csv("/home/claude/analysis/study1_cluster_robust.csv")

# =========================================================================
# FIG 1 — Study 1 distributional cues (already exists, keep as is)
# =========================================================================
# (unchanged, use existing fig_study1.png)

# =========================================================================
# FIG 2 — Paired Eng/Chi AoA for Choe 7 superordinates (bilingual bars)
# =========================================================================
fig, ax = plt.subplots(figsize=(7.4, 4.6))
choe_sup = bi[(bi["from_choe"]==True) & (bi["level"]=="Superordinate")].copy()
choe_sup = choe_sup.sort_values("Kup_AoA")
x = np.arange(len(choe_sup)); w = 0.38
be = ax.bar(x - w/2, choe_sup["Kup_AoA"], w, label="English (Kuperman et al., 2012)",
            color="#4C72B0", edgecolor="k")
bc = ax.bar(x + w/2, choe_sup["Xu_AoA"], w, label="Chinese (Xu et al., 2021)",
            color="#C44E52", edgecolor="k")
ax.set_xticks(x)
ax.set_xticklabels([f"{r['english']}\n{r['chinese']}" for _, r in choe_sup.iterrows()],
                    fontsize=10)
ax.set_ylabel("Mean subjective AoA (years)", fontsize=11)
ax.grid(True, axis="y", alpha=.3); ax.set_axisbelow(True)
for bars, col in [(be,"Kup_AoA"),(bc,"Xu_AoA")]:
    for rect, v in zip(bars, choe_sup[col]):
        if not np.isnan(v):
            ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.15,
                    f"{v:.1f}", ha="center", fontsize=9)
ax.legend(loc="upper left", fontsize=10, frameon=True)
ax.set_ylim(0, choe_sup[["Kup_AoA","Xu_AoA"]].max().max()*1.18)
plt.tight_layout()
plt.savefig("/home/claude/analysis/fig_bilingual_sup.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_bilingual_sup.png")

# =========================================================================
# FIG 3 — Per-domain super-vs-basic diff in z-scores, both languages
# =========================================================================
# Use z-scores (within-language standardisation)
xu_full = pd.read_excel("/mnt/project/2021Xu.xlsx")
kup = pd.read_excel("/mnt/project/2012KupermanAoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx")
xu_m, xu_s = float(xu_full["AoA Mean"].mean()), float(xu_full["AoA Mean"].std())
kup_m, kup_s = float(kup["AoARating.Mean"].mean()), float(kup["AoARating.Mean"].std())
bi["Xu_z"]  = (bi["Xu_AoA"]  - xu_m)  / xu_s
bi["Kup_z"] = (bi["Kup_AoA"] - kup_m) / kup_s
eng_p = bi.groupby(["domain","level"])["Kup_z"].mean().unstack()
chn_p = bi.groupby(["domain","level"])["Xu_z"].mean().unstack()
shared = sorted(set(eng_p.dropna().index) & set(chn_p.dropna().index))
eng = eng_p.loc[shared]; chn = chn_p.loc[shared]
diffs = pd.DataFrame({
    "English": eng["Superordinate"]-eng["Basic"],
    "Chinese": chn["Superordinate"]-chn["Basic"],
}).sort_values("English")

fig, ax = plt.subplots(figsize=(9.0, 5.4))
y = np.arange(len(diffs)); hw = 0.38
ax.barh(y - hw/2, diffs["English"], hw, label="English (Kuperman, 2012)",
        color="#4C72B0", edgecolor="k")
ax.barh(y + hw/2, diffs["Chinese"], hw, label="Chinese (Xu et al., 2021)",
        color="#C44E52", edgecolor="k")
ax.axvline(0, color="k", lw=1.2)
ax.set_yticks(y); ax.set_yticklabels(diffs.index, fontsize=10)
ax.set_xlabel("Mean AoA difference (z-scored within language):  "
              "Superordinate − Basic\n"
              "← superordinate earlier          basic earlier →", fontsize=10)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, axis="x", alpha=.3); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("/home/claude/analysis/fig_bilingual_diff.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_bilingual_diff.png")

# =========================================================================
# FIG 4 — Item-level scatter: bilingual correlation
# =========================================================================
fig, ax = plt.subplots(figsize=(6.6, 5.6))
use = bi.dropna(subset=["Kup_AoA","Xu_AoA"]).copy()
sup = use[use["level"]=="Superordinate"]
bas = use[use["level"]=="Basic"]
ax.scatter(sup["Kup_AoA"], sup["Xu_AoA"], s=130, marker="D",
           color="#4C72B0", edgecolor="k", label="Superordinate")
ax.scatter(bas["Kup_AoA"], bas["Xu_AoA"], s=55, marker="o",
           color="#DD8452", edgecolor="k", alpha=.8, label="Basic-level")
lims = [2, 14]
ax.plot(lims, lims, 'k--', alpha=.4, label="y = x (identity)")
from scipy.stats import linregress, spearmanr
r = linregress(use["Kup_AoA"], use["Xu_AoA"])
rho, _ = spearmanr(use["Kup_AoA"], use["Xu_AoA"])
xs = np.linspace(lims[0], lims[1], 50)
ax.plot(xs, r.slope*xs + r.intercept, color="grey", lw=1.2,
        label=f"Linear fit: r = {r.rvalue:.2f}, ρ = {rho:.2f}")
choe_sup = sup[sup["from_choe"]==True]
for _, row in choe_sup.iterrows():
    ax.annotate(row["english"], (row["Kup_AoA"], row["Xu_AoA"]),
                xytext=(5, 2), textcoords="offset points", fontsize=8)
ax.set_xlabel("English AoA (Kuperman et al., 2012), years", fontsize=11)
ax.set_ylabel("Chinese AoA (Xu et al., 2021), years", fontsize=11)
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.grid(True, alpha=.3); ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("/home/claude/analysis/fig_bilingual_scatter.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_bilingual_scatter.png")

# =========================================================================
# FIG 5 — NEW: Distribution → acquisition (critical tension)
# Two panels: (a) crit_index vs AoA, coloured by level;
#             (b) log_freq vs AoA
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

exp_ok = exp.dropna(subset=["AoA","crit_index"]).copy()
exp_ok["log_freq"] = np.log10(exp_ok["n_utt_Mandarin"].fillna(0) + 1)

sup = exp_ok[exp_ok["level"]=="Superordinate"]
bas = exp_ok[exp_ok["level"]=="Basic"]

# Panel a
ax1.scatter(bas["crit_index"], bas["AoA"], s=45, color="#DD8452",
            edgecolor="k", alpha=.7, label=f"Basic (n={len(bas)})")
ax1.scatter(sup["crit_index"], sup["AoA"], s=140, marker="D",
            color="#4C72B0", edgecolor="k", label=f"Super (n={len(sup)})")
from scipy.stats import linregress, spearmanr
r_all = linregress(exp_ok["crit_index"], exp_ok["AoA"])
rho_all, p_rho = spearmanr(exp_ok["crit_index"], exp_ok["AoA"])
xs = np.linspace(exp_ok["crit_index"].min(), exp_ok["crit_index"].max(), 50)
ax1.plot(xs, r_all.slope*xs + r_all.intercept, color="grey", lw=1.2, linestyle="--",
         label=f"overall ρ = {rho_all:+.2f}")
ax1.set_xlabel("Distributional-cue index (mean across 4 critical cues)", fontsize=10)
ax1.set_ylabel("Chinese AoA (Xu et al., 2021), years", fontsize=10)
ax1.set_title("(a) Distribution vs. AoA", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=.3); ax1.set_axisbelow(True)

# Panel b — frequency effect
ax2.scatter(bas["log_freq"], bas["AoA"], s=45, color="#DD8452",
            edgecolor="k", alpha=.7, label="Basic")
ax2.scatter(sup["log_freq"], sup["AoA"], s=140, marker="D",
            color="#4C72B0", edgecolor="k", label="Super")
r_f = linregress(exp_ok["log_freq"], exp_ok["AoA"])
rho_f, _ = spearmanr(exp_ok["log_freq"], exp_ok["AoA"])
xs = np.linspace(exp_ok["log_freq"].min(), exp_ok["log_freq"].max(), 50)
ax2.plot(xs, r_f.slope*xs + r_f.intercept, color="grey", lw=1.2, linestyle="--",
         label=f"overall ρ = {rho_f:+.2f}")
ax2.set_xlabel("log₁₀(1 + corpus frequency) in Mandarin CHILDES", fontsize=10)
ax2.set_ylabel("Chinese AoA (Xu et al., 2021), years", fontsize=10)
ax2.set_title("(b) Frequency vs. AoA", fontsize=11)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=.3); ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig("/home/claude/analysis/fig_distribution_vs_aoa.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_distribution_vs_aoa.png")

# =========================================================================
# FIG 6 — Cue-contribution comparison (from cluster-robust logistic)
# Kept from v1
# =========================================================================

# =========================================================================
# FIG 7 — Morphology (reduced — kept but with clearer framing)
# =========================================================================
fig, ax = plt.subplots(figsize=(6.6, 4.2))
choe = bi[bi["from_choe"]==True].copy()
counts = choe.groupby(["level","nChar"]).size().unstack(fill_value=0)
# order n_char as 1, 2, 3
counts = counts.reindex(columns=[1,2,3], fill_value=0)
counts.plot(kind="bar", stacked=True, ax=ax,
            color=["#8AB1D9","#4C72B0","#204F84"], edgecolor="k")
ax.set_ylabel("Number of items", fontsize=10)
ax.set_xlabel("Semantic level", fontsize=10)
ax.set_xticklabels(counts.index, rotation=0)
ax.legend(title="n. Chinese characters",
          labels=["1 (monomorphemic)","2 (disyllabic compound)","3 (longer)"])
ax.grid(True, axis="y", alpha=.3); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("/home/claude/analysis/fig_morphology.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_morphology.png")
