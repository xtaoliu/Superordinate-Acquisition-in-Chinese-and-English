"""
Regenerate target_counts.csv on the full 11-corpus dataset.
Gives a per-item breakdown with counts in each of the 11 corpora.
"""
import pandas as pd, re

df = pd.read_csv("adult_utterances_all.csv")
print(f"Adult utterances: {len(df):,}")
corpora = sorted(df["corpus"].unique().tolist())
print(f"Corpora ({len(corpora)}): {corpora}")

# Same target inventory as Study 1 (Choe items + Chinese-native substitutes)
TARGETS = [
    # (domain, level, Chinese, from_Choe, gloss)
    ("toy",       "Superordinate", "玩具",   True,  "toy"),
    ("animal",    "Superordinate", "动物",   True,  "animal"),
    ("tool",      "Superordinate", "工具",   True,  "tool"),
    ("building",  "Superordinate", "建筑",   True,  "building"),
    ("fruit",     "Superordinate", "水果",   True,  "fruit"),
    ("vegetable", "Superordinate", "蔬菜",   True,  "vegetable"),
    ("dessert",   "Superordinate", "甜点",   True,  "dessert"),
    ("toy",       "Basic",         "球",     True,  "ball (Choe main)"),
    ("toy",       "Basic",         "娃娃",   True,  "doll (Choe repl)"),
    ("toy",       "Basic",         "积木",   False, "building blocks"),
    ("animal",    "Basic",         "猫",     True,  "cat (Choe main)"),
    ("animal",    "Basic",         "熊",     True,  "bear (Choe repl)"),
    ("animal",    "Basic",         "狗",     False, "dog"),
    ("animal",    "Basic",         "鱼",     False, "fish"),
    ("animal",    "Basic",         "兔子",   False, "rabbit"),
    ("tool",      "Basic",         "叉子",   True,  "fork (Choe main)"),
    ("tool",      "Basic",         "筷子",   False, "chopsticks"),
    ("tool",      "Basic",         "勺子",   False, "spoon"),
    ("building",  "Basic",         "医院",   True,  "hospital (Choe main)"),
    ("building",  "Basic",         "房子",   False, "house"),
    ("building",  "Basic",         "学校",   False, "school"),
    ("fruit",     "Basic",         "草莓",   True,  "strawberry (Choe main)"),
    ("fruit",     "Basic",         "苹果",   False, "apple"),
    ("fruit",     "Basic",         "西瓜",   False, "watermelon"),
    ("vegetable", "Basic",         "辣椒",   True,  "pepper (Choe main)"),
    ("vegetable", "Basic",         "胡萝卜", True,  "carrot (Choe repl)"),
    ("vegetable", "Basic",         "西红柿", False, "tomato"),
    ("vegetable", "Basic",         "白菜",   False, "cabbage"),
    ("dessert",   "Basic",         "蛋糕",   False, "cake"),
    ("dessert",   "Basic",         "饼干",   False, "biscuit"),
]

def token_count(word, frame):
    """Count utterances containing word as a standalone space-delimited token."""
    pat = re.compile(rf"(^|[\s,，。.!?？！])({re.escape(word)})($|[\s,，。.!?？！])")
    return int(frame["utterance"].apply(
        lambda u: bool(pat.search(u)) if isinstance(u, str) else False
    ).sum())

rows = []
for domain, level, ch, from_choe, gloss in TARGETS:
    total = token_count(ch, df)
    rec = {"domain":domain, "level":level, "chinese":ch, "gloss":gloss,
           "from_choe":from_choe, "n_total":total}
    for c in corpora:
        rec[c] = token_count(ch, df[df["corpus"]==c])
    rows.append(rec)

tbl = pd.DataFrame(rows)
tbl.to_csv("target_counts_all.csv", index=False, encoding="utf-8-sig")

print("\n=== Per-item target counts across all 11 corpora ===")
# Compact view
view_cols = ["domain","level","chinese","gloss","from_choe","n_total"] + corpora
print(tbl[view_cols].to_string(index=False))

# Summary by level
print("\n=== Summary by semantic level ===")
print(tbl.groupby("level")["n_total"].agg(["count","sum","mean","median"]).round(1))
