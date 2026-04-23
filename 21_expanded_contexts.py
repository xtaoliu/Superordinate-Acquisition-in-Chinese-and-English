"""
Extract Mandarin distributional cues for the 122-item further-expanded set.
This provides a LARGER item-level dataset for reviewer-recommended regression.
"""
import pandas as pd, re, numpy as np

utt = pd.read_csv("/home/claude/analysis/adult_utterances_all.csv")
utt = utt[utt["utterance"].notna()].copy()

expanded = pd.read_csv("/home/claude/analysis/further_expanded.csv")
print(f"Expanded inventory: {len(expanded)} items")

CONTEXT_PATTERNS = {
    "quantifier":  re.compile(r"(一些|有些|所有|全部|每一|每个|每种|都\s)"),
    "whNP":        re.compile(r"(什么|哪\s*(个|些|种|类|只|条|本|块|张|头|根|样|位)?)"),
    "anchor":      re.compile(r"(一\s*种|一\s*类|哪\s*种|哪\s*类|种\s*类|类\s*型|几\s*种|这\s*种|那\s*种)"),
    "another":     re.compile(r"(另外|另一|另\s*一|别的|别\s*的|其他|其它)"),
    "labelling":   re.compile(r"(这|那|这个|那个|这些|那些)\s*(是|就是)\s*"),
}
def token_pat(t): return re.compile(rf"(^|[\s,，。.!?？！])({re.escape(t)})($|[\s,，。.!?？！])")
def definite_pat(t): return re.compile(rf"(这|那)\s*(个|些|种|只|条|本|块|张|头|根|样|位|把)?\s*{re.escape(t)}")

rows = []
for _, r in expanded.iterrows():
    ch = r["chinese"]
    tp = token_pat(ch); dp = definite_pat(ch)
    sub = utt[utt["utterance"].apply(lambda u: bool(tp.search(u)) if isinstance(u,str) else False)]
    n = len(sub)
    rec = dict(r)
    rec["n_utt_Mandarin"] = n
    if n == 0:
        for c in list(CONTEXT_PATTERNS.keys()) + ["definite","crit_index"]:
            rec[f"{c}_p"] = np.nan
        rows.append(rec)
        continue
    for c, pat in CONTEXT_PATTERNS.items():
        m = sub["utterance"].apply(lambda u: bool(pat.search(u)))
        rec[f"{c}_p"] = float(m.mean())
    rec["definite_p"] = float(sub["utterance"].apply(lambda u: bool(dp.search(u))).mean())
    rec["crit_index"] = float(np.mean([rec[f"{c}_p"] for c in ["quantifier","whNP","anchor","another"]]))
    rows.append(rec)

out = pd.DataFrame(rows)
out.to_csv("/home/claude/analysis/expanded_contexts.csv", index=False, encoding="utf-8-sig")
# Summary
print(f"\nItems with ≥ 1 corpus occurrence: {(out['n_utt_Mandarin']>0).sum()}/{len(out)}")
print(f"  n utterances summary: median={out['n_utt_Mandarin'].median():.0f}, "
      f"min={out['n_utt_Mandarin'].min()}, max={out['n_utt_Mandarin'].max()}")
print("\nCrit_index distribution by level:")
print(out.groupby("level")["crit_index"].describe().round(4))
print("\nn_utt by level:")
print(out.groupby("level")["n_utt_Mandarin"].describe().round(1))
