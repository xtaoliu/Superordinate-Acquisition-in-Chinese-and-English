"""
Build the final itemset. Per domain, we lock in the most-direct Chinese
translation of each English term from Choe & Papafragou (2026), choosing among
candidates by:  (a) it is the standard Mandarin gloss in modern dictionaries,
(b) it appears in at least one of the three databases.

Items absent from all three databases are recorded but excluded from analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Final, locked-in translation per (domain, level)
# - Primary set matches Choe (2026) Table 1 (main set)
# - Replication set matches Choe (2026) Table 4 (replication set)
ITEMS = [
    # (domain, set, level, English, Chinese)
    ("toy",       "main",        "Superordinate", "toy",        "玩具"),
    ("toy",       "main",        "Basic",         "ball",       "球"),
    ("toy",       "replication", "Basic",         "doll",       "娃娃"),

    ("animal",    "main",        "Superordinate", "animal",     "动物"),
    ("animal",    "main",        "Basic",         "cat",        "猫"),
    ("animal",    "replication", "Basic",         "bear",       "熊"),

    ("tool",      "main",        "Superordinate", "tool",       "工具"),
    ("tool",      "main",        "Basic",         "fork",       "叉子"),
    ("tool",      "replication", "Basic",         "hammer",     "锤子"),

    ("building",  "main",        "Superordinate", "building",   "建筑"),
    ("building",  "main",        "Basic",         "hospital",   "医院"),
    ("building",  "replication", "Basic",         "barn",       "谷仓"),

    ("fruit",     "main",        "Superordinate", "fruit",      "水果"),
    ("fruit",     "main",        "Basic",         "strawberry", "草莓"),
    ("fruit",     "replication", "Basic",         "kiwi",       "猕猴桃"),

    ("vegetable", "main",        "Superordinate", "vegetable",  "蔬菜"),
    ("vegetable", "main",        "Basic",         "pepper",     "辣椒"),
    ("vegetable", "replication", "Basic",         "carrot",     "胡萝卜"),

    ("dessert",   "main",        "Superordinate", "dessert",    "甜点"),
    ("dessert",   "main",        "Basic",         "waffle",     "华夫饼"),
    ("dessert",   "replication", "Basic",         "pancake",    "薄饼"),
]
items = pd.DataFrame(ITEMS,
    columns=["domain","set","level","english","chinese"])
items["nChar"] = items["chinese"].str.len()

# Databases
xu = pd.read_excel("/mnt/project/2021Xu.xlsx")
zc = pd.read_excel("/mnt/project/2023ZhangCharacter.xlsx", sheet_name="Data")
zc.columns = [c.strip() for c in zc.columns]
zw = pd.read_excel("/mnt/project/2023ZhangWord.xlsx",      sheet_name="Data")
zw.columns = [c.strip() for c in zw.columns]

# ------------------------------------------------------------------
# Enrich each item with every variable available across the 3 DBs
# ------------------------------------------------------------------
rows = []
for _, it in items.iterrows():
    ch, n = it["chinese"], it["nChar"]
    row = it.to_dict()

    # Xu 2021 (multi-char words)
    m = xu[xu["Word"] == ch]
    row["in_Xu"]       = len(m) > 0
    row["Xu_AoA"]      = float(m["AoA Mean"].iloc[0]) if len(m) else np.nan
    row["Xu_AoA_SD"]   = float(m["AoA SD"].iloc[0])   if len(m) else np.nan
    row["Xu_Raters"]   = int  (m["No. of Raters"].iloc[0]) if len(m) else np.nan

    # Zhang 2023 character (single char only)
    m = zc[zc["Char"] == ch]
    row["in_ZC"]          = len(m) > 0
    for v in ["Volume","nStroke","nRadical","nPronunciation",
              "nMeaning","Count_Sum","logCHR-CD","RTs_A","RTs_Y","RTs_O",
              "ACC_A","ACC_Y","ACC_O"]:
        row[f"ZC_{v}"] = float(m[v].iloc[0]) if len(m) else np.nan

    # Zhang 2023 word (multi-char words)
    m = zw[zw["Word"] == ch]
    row["in_ZW"]          = len(m) > 0
    for v in ["Volume","Length","nPronunciation","nMeaning","Count_Sum",
              "logW-CD","Sum_nStroke","Sum_nMeaning",
              "RTs_A","RTs_Y","RTs_O","ACC_A","ACC_Y","ACC_O"]:
        row[f"ZW_{v}"] = float(m[v].iloc[0]) if len(m) else np.nan
    rows.append(row)

enriched = pd.DataFrame(rows)
out = Path("/home/claude/analysis")
enriched.to_csv(out / "items_enriched.csv", index=False, encoding="utf-8-sig")

# Quick readable view
view_cols = ["domain","set","level","english","chinese","nChar",
             "in_Xu","Xu_AoA","in_ZC","ZC_Volume","in_ZW","ZW_Volume"]
print(enriched[view_cols].to_string(index=False))
