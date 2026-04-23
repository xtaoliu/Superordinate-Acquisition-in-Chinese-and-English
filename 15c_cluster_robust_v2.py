"""
Cluster-robust logistic regression, fitted via IRLS with ridge, then compute
cluster-robust SE clustered by item.  
"""
import pandas as pd, numpy as np
from scipy import stats

per_utt = pd.read_csv("study1_per_utterance_all.csv")
per_utt["is_super"] = (per_utt["level"]=="Superordinate").astype(int)

CONTEXTS = ["quantifier","whNP","anchor","another","labelling","definite"]
CTX_LABEL = {"quantifier":"some/all","whNP":"what/which",
             "anchor":"kind/type","another":"another/other",
             "labelling":"this/that+BE","definite":"this/that+N"}
IS_CRIT = {"quantifier":True,"whNP":True,"anchor":True,"another":True,
           "labelling":False,"definite":False}

def fit_logit_irls(y, X, max_iter=100, tol=1e-8):
    """IRLS with step-halving."""
    n, k = X.shape
    beta = np.zeros(k)
    # init with log odds of y
    p_init = (y.sum() + 0.5)/(len(y) + 1)
    beta[0] = np.log(p_init/(1-p_init))
    ll_prev = -np.inf
    for it in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1 - p)
        w = np.maximum(w, 1e-10)
        z = eta + (y - p) / w
        WX = X * w[:, None]
        try:
            H = X.T @ WX
            XWz = X.T @ (w * z)
            beta_new = np.linalg.solve(H + 1e-8*np.eye(k), XWz)
        except np.linalg.LinAlgError:
            break
        # step-halving if LL decreases
        def ll(b):
            e = np.clip(X @ b, -30, 30)
            return np.sum(y*e - np.logaddexp(0, e))
        ll_new = ll(beta_new)
        t = 1.0
        while ll_new < ll_prev - 1e-6 and t > 1e-6:
            t *= 0.5
            beta_new = beta + t * (beta_new - beta)
            ll_new = ll(beta_new)
        if abs(ll_new - ll_prev) < tol: 
            beta = beta_new; break
        beta = beta_new
        ll_prev = ll_new
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0/(1.0+np.exp(-eta))
    return beta, p

def cluster_robust_cov(X, y, p, clusters):
    n, k = X.shape
    resid = y - p
    w = p * (1 - p); w = np.maximum(w, 1e-10)
    B_inv = (X.T * w) @ X + 1e-8*np.eye(k)
    B = np.linalg.inv(B_inv)
    score = X * resid[:, None]
    M = np.zeros((k, k))
    for cl in np.unique(clusters):
        m = clusters == cl
        sc = score[m].sum(axis=0)
        M += np.outer(sc, sc)
    G = len(np.unique(clusters))
    adj = G/(G-1) * (n-1)/(n-k)
    return adj * B @ M @ B

print(f"\n{'Context':14}  {'β':>8}  {'SE_cr':>7}   {'OR':>6}  {'95%CI':>18}  {'z':>6}  {'p_CR':>8}  {'type':>9}")
print("-"*90)
out_rows = []
for c in CONTEXTS:
    y = per_utt[c].astype(float).values
    is_sup = per_utt["is_super"].astype(float).values
    X = np.column_stack([np.ones(len(y)), is_sup])
    beta, p_hat = fit_logit_irls(y, X)
    cov_cr = cluster_robust_cov(X, y, p_hat, per_utt["chinese"].values)
    b = beta[1]
    se_cr = np.sqrt(cov_cr[1,1])
    OR = np.exp(b)
    lo = np.exp(b - 1.96*se_cr); hi = np.exp(b + 1.96*se_cr)
    z = b/se_cr
    p = 2*(1 - stats.norm.cdf(abs(z)))
    print(f"{CTX_LABEL[c]:14}  {b:>8.3f}  {se_cr:>7.3f}  {OR:>6.2f}  "
          f"[{lo:>5.2f}, {hi:>5.2f}]  {z:>6.2f}  {p:>8.4f}  "
          f"{'critical' if IS_CRIT[c] else 'control':>9}")
    out_rows.append({"context":c,"beta":b,"se_cluster":se_cr,"OR":OR,
                     "lo_OR":lo,"hi_OR":hi,"z":z,"p":p,
                     "type":"critical" if IS_CRIT[c] else "control"})

pd.DataFrame(out_rows).to_csv("study1_cluster_robust.csv", index=False, encoding="utf-8-sig")

# =====================================================================
# ALSO: random-effects logistic regression for COMPARISON via the
# Agresti-Coull approach: compute per-item proportion, then DeLong-style
# aggregated test. This gives an ITEM-LEVEL comparison.
# =====================================================================
print("\n--- Item-level comparison (each item's proportion as unit of analysis) ---")
res = pd.read_csv("study1_context_counts_all.csv")
print(f"{'Context':14} {'n_items_sup':>12} {'n_items_basic':>14} {'mean_p_sup':>10} {'mean_p_basic':>12} {'t':>6} {'p':>8}")
print("-"*84)
for c in CONTEXTS:
    sup = res.loc[res.level=='Superordinate', f'{c}_p']
    bas = res.loc[res.level=='Basic',         f'{c}_p']
    t,p = stats.ttest_ind(sup, bas, equal_var=False)
    print(f"{CTX_LABEL[c]:14} {len(sup):>12} {len(bas):>14} {sup.mean():>10.4f} "
          f"{bas.mean():>12.4f} {t:>6.2f} {p:>8.4f}")
