"""
Study 1 (replication on Chinese): Extract the superordinate/basic-level noun
pairs from Choe & Papafragou (2026), Tables 1 and 4, and match them against
the Chinese lexical databases from Xu et al. (2021) and Zhang et al. (2023).

Open-science note: every translation decision is explicit, every hit and miss
is logged, and nothing is silently dropped.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# Step 1. Choe & Papafragou (2026) noun pairs, with Chinese glosses.
# Glosses chosen from Modern Standard Mandarin; where more than one
# candidate exists we record both and let database membership decide.
# ------------------------------------------------------------------
PAIRS = [
    # domain,         level,           English,      Chinese candidates
    ("toy",           "Superordinate", "toy",        ["玩具"]),
    ("toy",           "Basic_main",    "ball",       ["球"]),
    ("toy",           "Basic_repl",    "doll",       ["娃娃", "玩偶"]),

    ("animal",        "Superordinate", "animal",     ["动物"]),
    ("animal",        "Basic_main",    "cat",        ["猫"]),
    ("animal",        "Basic_repl",    "bear",       ["熊"]),

    ("tool",          "Superordinate", "tool",       ["工具"]),
    ("tool",          "Basic_main",    "fork",       ["叉子"]),
    ("tool",          "Basic_repl",    "hammer",     ["锤子", "榔头"]),

    ("building",      "Superordinate", "building",   ["建筑", "楼房"]),
    ("building",      "Basic_main",    "hospital",   ["医院"]),
    ("building",      "Basic_repl",    "barn",       ["谷仓", "粮仓"]),

    ("fruit",         "Superordinate", "fruit",      ["水果"]),
    ("fruit",         "Basic_main",    "strawberry", ["草莓"]),
    ("fruit",         "Basic_repl",    "kiwi",       ["猕猴桃", "奇异果"]),

    ("vegetable",     "Superordinate", "vegetable",  ["蔬菜"]),
    ("vegetable",     "Basic_main",    "pepper",     ["辣椒", "胡椒"]),
    ("vegetable",     "Basic_repl",    "carrot",     ["胡萝卜", "萝卜"]),

    ("dessert",       "Superordinate", "dessert",    ["甜点", "点心", "甜品"]),
    ("dessert",       "Basic_main",    "waffle",     ["华夫饼", "松饼"]),
    ("dessert",       "Basic_repl",    "pancake",    ["煎饼", "薄饼"]),
]

# Single-character basic-level items that will ONLY appear in Zhang Character
SINGLE_CHAR_BASICS = ["球", "猫", "熊"]

# ------------------------------------------------------------------
# Step 2. Load datasets
# ------------------------------------------------------------------
xu = pd.read_excel("/mnt/project/2021Xu.xlsx")
zc = pd.read_excel("/mnt/project/2023ZhangCharacter.xlsx", sheet_name="Data")
zw = pd.read_excel("/mnt/project/2023ZhangWord.xlsx", sheet_name="Data")

print(f"Xu 2021 (words):            {xu.shape[0]:>6} rows")
print(f"Zhang 2023 (characters):    {zc.shape[0]:>6} rows")
print(f"Zhang 2023 (words):         {zw.shape[0]:>6} rows")

# Normalize whitespace in column names for Zhang characters (has "nMeaning ")
zc.columns = [c.strip() for c in zc.columns]
zw.columns = [c.strip() for c in zw.columns]

# ------------------------------------------------------------------
# Step 3. Look up every candidate Chinese form in each dataset
# ------------------------------------------------------------------
records = []
for domain, level, english, candidates in PAIRS:
    for cand in candidates:
        in_xu  = xu [xu ["Word"] == cand]
        in_zc  = zc [zc ["Char"] == cand]        # single-char items
        in_zw  = zw [zw ["Word"] == cand]        # multi-char words
        records.append({
            "domain"      : domain,
            "level"       : level,
            "english"     : english,
            "chinese"     : cand,
            "xu_found"    : len(in_xu)  > 0,
            "zc_found"    : len(in_zc)  > 0,
            "zw_found"    : len(in_zw)  > 0,
            "xu_aoa"      : float(in_xu ["AoA Mean"].iloc[0]) if len(in_xu) else np.nan,
            "zc_volume"   : float(in_zc ["Volume"  ].iloc[0]) if len(in_zc) else np.nan,
            "zw_volume"   : float(in_zw ["Volume"  ].iloc[0]) if len(in_zw) else np.nan,
        })
lookup = pd.DataFrame(records)

out = Path("/home/claude/analysis")
lookup.to_csv(out / "lookup_table.csv", index=False, encoding="utf-8-sig")

print("\n=== Lookup results (Choe 2026 pairs in Chinese databases) ===")
print(lookup.to_string(index=False))

# ------------------------------------------------------------------
# Step 4. Coverage summary
# ------------------------------------------------------------------
print("\n=== Coverage summary ===")
print(f"Any match in at least one DB : {((lookup[['xu_found','zc_found','zw_found']].any(axis=1))).sum()} / {len(lookup)}")
print(f"In Xu 2021                    : {lookup['xu_found'].sum()}")
print(f"In Zhang 2023 characters      : {lookup['zc_found'].sum()}")
print(f"In Zhang 2023 words           : {lookup['zw_found'].sum()}")
