"""
Run context extraction on the expanded 11-corpus Mandarin CHILDES dataset.
Identical methodology to Study 1 v1 (file 10_extract_contexts.py), now with
more data.
"""
import pandas as pd, re

df = pd.read_csv("adult_utterances_all.csv")
df = df[df["utterance"].notna()].copy()
print(f"Adult utterances: {len(df):,}")

TARGETS = [
    ("toy",       "Superordinate", "玩具",   True,  "toy"),
    ("animal",    "Superordinate", "动物",   True,  "animal"),
    ("tool",      "Superordinate", "工具",   True,  "tool"),
    ("building",  "Superordinate", "建筑",   True,  "building"),
    ("fruit",     "Superordinate", "水果",   True,  "fruit"),
    ("vegetable", "Superordinate", "蔬菜",   True,  "vegetable"),
    ("dessert",   "Superordinate", "甜点",   True,  "dessert"),
    ("toy",       "Basic",         "球",     True,  "ball (Choe main)"),
    ("toy",       "Basic",         "娃娃",   True,  "doll (Choe repl)"),
    ("animal",    "Basic",         "猫",     True,  "cat (Choe main)"),
    ("animal",    "Basic",         "熊",     True,  "bear (Choe repl)"),
    ("tool",      "Basic",         "叉子",   True,  "fork (Choe main)"),
    ("building",  "Basic",         "医院",   True,  "hospital (Choe main)"),
    ("fruit",     "Basic",         "草莓",   True,  "strawberry (Choe main)"),
    ("vegetable", "Basic",         "辣椒",   True,  "pepper (Choe main)"),
    ("vegetable", "Basic",         "胡萝卜", True,  "carrot (Choe repl)"),
    ("toy",       "Basic",         "积木",   False, "building blocks"),
    ("animal",    "Basic",         "狗",     False, "dog"),
    ("animal",    "Basic",         "鱼",     False, "fish"),
    ("animal",    "Basic",         "兔子",   False, "rabbit"),
    ("tool",      "Basic",         "筷子",   False, "chopsticks"),
    ("tool",      "Basic",         "勺子",   False, "spoon"),
    ("building",  "Basic",         "房子",   False, "house"),
    ("building",  "Basic",         "学校",   False, "school"),
    ("fruit",     "Basic",         "苹果",   False, "apple"),
    ("fruit",     "Basic",         "西瓜",   False, "watermelon"),
    ("vegetable", "Basic",         "西红柿", False, "tomato"),
    ("vegetable", "Basic",         "白菜",   False, "cabbage"),
    ("dessert",   "Basic",         "蛋糕",   False, "cake"),
    ("dessert",   "Basic",         "饼干",   False, "biscuit"),
]

CONTEXT_PATTERNS = {
    "quantifier":  re.compile(r"(一些|有些|所有|全部|每一|每个|每种|都\s)"),
    "whNP":        re.compile(r"(什么|哪\s*(个|些|种|类|只|条|本|块|张|头|根|样|位)?)"),
    "anchor":      re.compile(r"(一\s*种|一\s*类|哪\s*种|哪\s*类|种\s*类|类\s*型|几\s*种|这\s*种|那\s*种)"),
    "another":     re.compile(r"(另外|另一|另\s*一|别的|别\s*的|其他|其它)"),
    "labelling":   re.compile(r"(这|那|这个|那个|这些|那些)\s*(是|就是)\s*"),
    "definite":    None,
}

def definite_pat(t): return re.compile(rf"(这|那)\s*(个|些|种|只|条|本|块|张|头|根|样|位|把)?\s*{re.escape(t)}")
def token_pat(t):    return re.compile(rf"(^|[\s,，。.!?？！])({re.escape(t)})($|[\s,，。.!?？！])")

rows = []
per_utt_rows = []  # for mixed-effects model
for domain, level, ch, from_choe, gloss in TARGETS:
    tp = token_pat(ch); dp = definite_pat(ch)
    sub = df[df["utterance"].str.contains(ch, na=False, regex=False)].copy()
    sub = sub[sub["utterance"].apply(lambda u: bool(tp.search(u)) if isinstance(u,str) else False)]
    n_utt = len(sub)
    if n_utt == 0: continue
    rec = {"domain":domain,"level":level,"chinese":ch,"gloss":gloss,
           "from_choe":from_choe,"n_utterances":n_utt}
    for ctx, pat in CONTEXT_PATTERNS.items():
        if ctx == "definite":
            matched = sub["utterance"].apply(lambda u: bool(dp.search(u)))
        else:
            matched = sub["utterance"].apply(lambda u: bool(pat.search(u)) if isinstance(u,str) else False)
        rec[f"{ctx}_n"] = int(matched.sum())
        rec[f"{ctx}_p"] = float(matched.mean())
    rows.append(rec)
    # also store per-utterance rows for mixed-effects
    for idx, row in sub.iterrows():
        rec_u = {"corpus": row["corpus"], "file": row["file"],
                 "chinese": ch, "level": level, "domain": domain, "from_choe": from_choe}
        for ctx, pat in CONTEXT_PATTERNS.items():
            if ctx == "definite":
                rec_u[ctx] = int(bool(dp.search(row["utterance"])))
            else:
                rec_u[ctx] = int(bool(pat.search(row["utterance"])))
        per_utt_rows.append(rec_u)

results = pd.DataFrame(rows)
per_utt = pd.DataFrame(per_utt_rows)
results.to_csv("study1_context_counts_all.csv", index=False, encoding="utf-8-sig")
per_utt.to_csv("study1_per_utterance_all.csv", index=False, encoding="utf-8-sig")

print("\n=== Target inventory: n_utterances per item ===")
print(results[["domain","level","chinese","gloss","from_choe","n_utterances"]].to_string(index=False))
print(f"\nTotal target-utterance tokens: {int(results['n_utterances'].sum()):,}")
print(f"  Superordinate: {int(results[results.level=='Superordinate'].n_utterances.sum()):,}")
print(f"  Basic:         {int(results[results.level=='Basic'].n_utterances.sum()):,}")
