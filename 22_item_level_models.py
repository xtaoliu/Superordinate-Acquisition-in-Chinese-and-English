"""
Item-level regression models on the 122-item expanded set.
Addresses reviewer's core request (Section A) + frequency control (D) +
morphology claim (C) + separate item sets (E).
"""
import pandas as pd, numpy as np
from scipy import stats

# Load expanded set with crit_index + Xu AoA
exp = pd.read_csv("/home/claude/analysis/expanded_contexts.csv")
# Mandarin corpus frequency (log)
exp["log_freq_cn"] = np.log10(exp["n_utt_Mandarin"].fillna(0) + 1)
# Drop items with no corpus occurrence or no Xu AoA
d = exp.dropna(subset=["crit_index","AoA"]).copy()
d["is_super"] = (d["level"]=="Superordinate").astype(int)
print(f"Items with Xu AoA AND crit_index: {len(d)}")
print(f"  Super: {d['is_super'].sum()}, Basic: {(1-d['is_super']).sum()}")

# Add Kuperman English AoA via merge on english name — need english field
bi = pd.read_csv("/home/claude/analysis/bilingual_xu_kup.csv")
# bi has items only for the Choe set and the 8 added domains' super/sample-basic
# We need a broader English AoA — look up every chinese in our expanded set
# Not all expanded items have English translations in bi, so fill what we can:
en_map = bi[["chinese","english","Kup_AoA"]].dropna(subset=["english"]).drop_duplicates("chinese")
d = d.merge(en_map, on="chinese", how="left")

# -------------------------------------------------------------------------
def ols_with_se(y, X, names):
    n, k = X.shape
    b = np.linalg.solve(X.T @ X + 1e-8*np.eye(k), X.T @ y)
    resid = y - X @ b
    df_resid = n - k
    sigma2 = (resid @ resid) / df_resid
    cov = sigma2 * np.linalg.inv(X.T @ X + 1e-8*np.eye(k))
    se = np.sqrt(np.diag(cov))
    t_stats = b / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_resid))
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (resid @ resid)
    r2 = 1 - ss_res/ss_tot
    r2_adj = 1 - (1-r2) * (n-1) / (n-k)
    # AIC
    ll = -0.5 * n * (np.log(2*np.pi) + np.log(ss_res/n) + 1)
    aic = -2*ll + 2*k
    return pd.DataFrame({"coef":b,"se":se,"t":t_stats,"p":p_vals}, index=names), r2, r2_adj, aic

def design(df, cols):
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    return X, ["(Intercept)"] + cols

def hr(t): print("\n"+"="*78+f"\n {t}\n"+"="*78)

# -------------------------------------------------------------------------
# Strict set: Choe items only (n=14 with both data)
# Expanded set: all 122 items with Xu + cues (n≈115)
# -------------------------------------------------------------------------
hr("A. Core reviewer analysis — CHINESE AoA ~ distribution + freq + morphology + level")

# Model 0: AoA ~ level only (baseline Choe prediction)
X0, n0 = design(d, ["is_super"])
r0 = ols_with_se(d["AoA"].values, X0, n0)
print(f"\nM0: AoA ~ is_super  |  N={len(d)}  R²={r0[1]:.3f}  AIC={r0[3]:.1f}")
print(r0[0].round(3))

# Model 1: AoA ~ crit_index  (does distribution predict acquisition?)
X1, n1 = design(d, ["crit_index"])
r1 = ols_with_se(d["AoA"].values, X1, n1)
print(f"\nM1: AoA ~ crit_index  |  R²={r1[1]:.3f}  AIC={r1[3]:.1f}")
print(r1[0].round(3))

# Model 2: AoA ~ crit_index + log_freq_cn  (add frequency, reviewer D)
X2, n2 = design(d, ["crit_index","log_freq_cn"])
r2 = ols_with_se(d["AoA"].values, X2, n2)
print(f"\nM2: AoA ~ crit_index + log_freq_cn  |  R²={r2[1]:.3f}  AIC={r2[3]:.1f}")
print(r2[0].round(3))

# Model 3: + is_super
X3, n3 = design(d, ["crit_index","log_freq_cn","is_super"])
r3 = ols_with_se(d["AoA"].values, X3, n3)
print(f"\nM3: AoA ~ crit_index + log_freq_cn + is_super  |  R²={r3[1]:.3f}  AIC={r3[3]:.1f}")
print(r3[0].round(3))

# Model 4: interaction crit_index × is_super (reviewer Add4)
d["crit_x_super"] = d["crit_index"] * d["is_super"]
X4, n4 = design(d, ["crit_index","log_freq_cn","is_super","crit_x_super"])
r4 = ols_with_se(d["AoA"].values, X4, n4)
print(f"\nM4: + crit_index × is_super  |  R²={r4[1]:.3f}  AIC={r4[3]:.1f}")
print(r4[0].round(3))

# Save all model outputs
for i, res in enumerate([r0,r1,r2,r3,r4]):
    df_out = res[0].copy()
    df_out["R2"] = res[1]
    df_out["R2_adj"] = res[2]
    df_out["AIC"] = res[3]
    df_out.to_csv(f"/home/claude/analysis/model_cn_M{i}.csv", encoding="utf-8-sig")

# -------------------------------------------------------------------------
# Parallel in English: Kuperman AoA ~ Mandarin distribution
# -------------------------------------------------------------------------
hr("A. Parallel: ENGLISH AoA ~ Mandarin distribution (for items with both)")
de = d.dropna(subset=["Kup_AoA"]).copy()
de["log_freq_cn"] = np.log10(de["n_utt_Mandarin"].fillna(0) + 1)
print(f"N with Kup AoA AND Mandarin crit_index: {len(de)}")

if len(de) >= 8:
    X, names = design(de, ["crit_index","log_freq_cn","is_super"])
    re_out = ols_with_se(de["Kup_AoA"].values, X, names)
    print(f"\nKup AoA ~ Mandarin crit_index + log_freq_cn + is_super  "
          f"|  R²={re_out[1]:.3f}  AIC={re_out[3]:.1f}")
    print(re_out[0].round(3))

# -------------------------------------------------------------------------
# Strict translation set only (Choe items), repeat analyses
# -------------------------------------------------------------------------
hr("E. Strict-translation set (Choe 2026 items only)")
bi_c = bi[bi["from_choe"]==True].copy()
# Merge crit_index from expanded_contexts
bi_c = bi_c.merge(
    exp[["chinese","crit_index","n_utt_Mandarin","log_freq_cn" if False else "crit_index"]].drop_duplicates("chinese")[["chinese","crit_index","n_utt_Mandarin"]],
    on="chinese", how="left"
).drop_duplicates("chinese")
# Use already-computed Mandarin crit_index if present; compute log_freq
bi_c["log_freq_cn"] = np.log10(bi_c["n_utt_Mandarin"].fillna(0) + 1)
bi_c = bi_c.dropna(subset=["Xu_AoA","crit_index"]).copy()
bi_c["is_super"] = (bi_c["level"]=="Superordinate").astype(int)
print(f"N Choe items with both Xu AoA AND distribution: {len(bi_c)}")
if len(bi_c) >= 5:
    X, names = design(bi_c, ["crit_index","log_freq_cn","is_super"])
    rc = ols_with_se(bi_c["Xu_AoA"].values, X, names)
    print(f"\nChoe-only model: AoA ~ crit_index + log_freq + is_super  "
          f"|  R²={rc[1]:.3f}  AIC={rc[3]:.1f}")
    print(rc[0].round(3))

# -------------------------------------------------------------------------
# Mixed-effects-lite: random intercept by domain, REML via simple iteration
# -------------------------------------------------------------------------
hr("B2. Random-intercept model: AoA ~ crit_index + log_freq + is_super + (1|domain)")
# Simple two-step: fit OLS, then estimate between-domain variance, then GLS with that variance
# This is a simplified REML.
def fit_random_intercept(y, X, groups, max_iter=50, tol=1e-6):
    """AoA ~ X + u_j where u_j ~ N(0, sigma_u²).
    Returns: beta, sigma_e², sigma_u², SE(beta)"""
    n, k = X.shape
    unique_g = np.unique(groups)
    ng = len(unique_g)
    # initial: OLS
    b = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ b
    sigma_e2 = float((resid @ resid) / (n - k))
    sigma_u2 = sigma_e2 * 0.25  # init
    for _ in range(max_iter):
        # V = sigma_e² I + sigma_u² Z Z'  where Z is indicator of group
        # For panel data with group sizes n_j: V block-diagonal
        # Work group-by-group to avoid constructing full V
        XtVinvX = np.zeros((k,k))
        XtVinvy = np.zeros(k)
        ll_terms = 0.0
        sum_u = 0.0
        sum_u2 = 0.0
        group_preds = {}
        for g in unique_g:
            m = groups == g
            Xg = X[m]; yg = y[m]
            nj = int(m.sum())
            # V_j = sigma_e² I_nj + sigma_u² J_nj  where J is ones matrix
            # V_j^{-1} = (1/sigma_e²) (I - sigma_u²/(sigma_e² + nj*sigma_u²) J)
            a = sigma_u2 / (sigma_e2 + nj * sigma_u2) / sigma_e2
            Vinv = (np.eye(nj) / sigma_e2) - a * np.ones((nj,nj))
            XtVinvX += Xg.T @ Vinv @ Xg
            XtVinvy += Xg.T @ Vinv @ yg
        b_new = np.linalg.solve(XtVinvX, XtVinvy)
        # Update variance components via MoM
        resid_new = y - X @ b_new
        # Compute sums within groups
        ss_within = 0.0; ss_between = 0.0
        for g in unique_g:
            m = groups == g
            rg = resid_new[m]
            ss_within += ((rg - rg.mean())**2).sum()
        # sigma_e² ≈ within-group mean squared error
        sigma_e2_new = max(ss_within / (n - ng), 1e-6)
        # sigma_u² ≈ (between-group MS - sigma_e²) / mean(n_j)
        group_means = np.array([resid_new[groups==g].mean() for g in unique_g])
        n_per = np.array([sum(groups==g) for g in unique_g])
        ss_between = (n_per * (group_means**2)).sum()
        ms_between = ss_between / (ng - 1) if ng > 1 else 0
        n_mean = float(n_per.mean())
        sigma_u2_new = max((ms_between - sigma_e2_new) / n_mean, 0)
        if (abs(b_new - b).max() < tol and
            abs(sigma_e2_new - sigma_e2) < tol and
            abs(sigma_u2_new - sigma_u2) < tol):
            b = b_new; sigma_e2 = sigma_e2_new; sigma_u2 = sigma_u2_new
            break
        b = b_new; sigma_e2 = sigma_e2_new; sigma_u2 = sigma_u2_new
    cov = np.linalg.inv(XtVinvX)
    return b, sigma_e2, sigma_u2, np.sqrt(np.diag(cov))

X, names = design(d, ["crit_index","log_freq_cn","is_super"])
b, se2, su2, se = fit_random_intercept(d["AoA"].values, X, d["domain"].values)
print(f"\nVariance components: σ²_e = {se2:.3f}, σ²_u(domain) = {su2:.3f}")
icc = su2 / (su2 + se2) if (su2+se2) > 0 else 0
print(f"ICC(domain) = {icc:.3f}")
print(f"\n{'Predictor':<20} {'coef':>10} {'SE':>8} {'t':>7} {'p':>8}")
for name, coef, s in zip(names, b, se):
    df_approx = len(d) - len(names)
    t = coef / s if s > 0 else 0
    p = 2 * (1 - stats.t.cdf(abs(t), df=df_approx))
    print(f"{name:<20} {coef:>10.3f} {s:>8.3f} {t:>7.3f} {p:>8.4f}")

# -------------------------------------------------------------------------
# Add5: Logistic classifier — can cues predict level (sufficiency)
# -------------------------------------------------------------------------
hr("Add5. Logistic classifier — are the 4 critical cues jointly sufficient?")
dc = exp.dropna(subset=["quantifier_p","whNP_p","anchor_p","another_p"]).copy()
dc["is_super"] = (dc["level"]=="Superordinate").astype(int)
y = dc["is_super"].values.astype(float)
X = np.column_stack([np.ones(len(dc)),
                     dc["quantifier_p"].values, dc["whNP_p"].values,
                     dc["anchor_p"].values, dc["another_p"].values])

def fit_logit_irls(y, X, max_iter=200):
    n, k = X.shape
    b = np.zeros(k)
    p0 = (y.sum()+0.5)/(len(y)+1)
    b[0] = np.log(p0/(1-p0))
    for _ in range(max_iter):
        eta = np.clip(X @ b, -30, 30)
        p = 1/(1+np.exp(-eta))
        w = np.maximum(p*(1-p), 1e-10)
        z = eta + (y-p)/w
        WX = X * w[:,None]
        try:
            b_new = np.linalg.solve(X.T @ WX + 1e-6*np.eye(k), X.T @ (w*z))
        except np.linalg.LinAlgError:
            break
        if np.abs(b_new - b).max() < 1e-8: 
            b = b_new; break
        b = b_new
    return b

b = fit_logit_irls(y, X)
eta = X @ b; p_hat = 1/(1+np.exp(-eta))
correct = ((p_hat > 0.5).astype(int) == y).mean()
from scipy.stats import mannwhitneyu
u, _ = mannwhitneyu(p_hat[y==1], p_hat[y==0], alternative="two-sided")
auc = u / (sum(y==1) * sum(y==0))
# Cluster-robust-ish SE (domain as clusters)
groups = dc["domain"].values
w = p_hat*(1-p_hat); w = np.maximum(w, 1e-10)
B_inv = (X.T * w) @ X + 1e-6*np.eye(X.shape[1])
B = np.linalg.inv(B_inv)
score = X * (y - p_hat)[:,None]
M = np.zeros((X.shape[1], X.shape[1]))
for g in np.unique(groups):
    m = groups == g
    sc = score[m].sum(axis=0)
    M += np.outer(sc, sc)
G = len(np.unique(groups))
n = len(y); k = X.shape[1]
adj = G/(G-1) * (n-1)/(n-k)
cov_cr = adj * B @ M @ B
se = np.sqrt(np.diag(cov_cr))

print(f"\nFitted logistic classifier (is_super ~ 4 critical cues) on {len(dc)} items")
print(f"  Classification accuracy at .5: {correct:.3f}")
print(f"  AUC (Mann-Whitney-derived):    {auc:.3f}")
print(f"\n{'Cue':<12} {'coef':>9} {'SE_cr':>8} {'OR':>8} {'z':>7} {'p':>8}")
for name, coef, s in zip(["(Int)","quant","whNP","anchor","another"], b, se):
    z = coef/s if s > 0 else 0
    p = 2*(1 - stats.norm.cdf(abs(z)))
    print(f"{name:<12} {coef:>+9.3f} {s:>8.3f} {np.exp(coef):>8.3f} {z:>+7.2f} {p:>8.4f}")

# Save
out = pd.DataFrame({
    "cue":["intercept","quantifier","whNP","anchor","another"],
    "coef":b, "se_cr":se, "OR":np.exp(b),
    "z": b/np.where(se>0, se, np.nan),
    "p": 2*(1 - stats.norm.cdf(np.abs(b/np.where(se>0, se, np.nan))))
})
out.to_csv("/home/claude/analysis/classifier_level_from_cues.csv", index=False, encoding="utf-8-sig")
