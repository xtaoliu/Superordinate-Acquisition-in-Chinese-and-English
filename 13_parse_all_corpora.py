"""
Parse all 11 Mandarin CHILDES corpora. Expands from 3 to 11 corpora.

Corpora:
  BJCMC           — Beijing Child Mandarin (49 files, Beijing)
  ChangPlay       — Chang toy-play (56 files, Taiwan)
  Erbaugh         — Erbaugh longitudinal (64 files, Taiwan)
  NSCtoys         — National Science Council toys (626 files, Taiwan)
  TCCM            — Taiwan Corpus of Child Mandarin (126 files, Taiwan)
  Tong            — Tong longitudinal (22 files, Shenzhen)
  Zhou1/2/3       — Zhou corpora (50 + 140 + 30 files, Shanghai/Nanjing)
  ZhouAssessment  — Zhou assessment (219 files, multi-site)
  ZhouDinner      — Zhou dinner-table (71 files, Shanghai)
"""
import re, csv
from pathlib import Path

CORPORA = [
    "BJCMC", "ChangPlay", "Erbaugh", "NSCtoys", "TCCM",
    "Tong", "Zhou1", "Zhou2", "Zhou3",
    "ZhouAssessment", "ZhouDinner"
]
ROOT = Path("/home/claude/corpora")

NON_ADULT = {"CHI","SIS","BRO","TAR","CH1","CH2","CH3","BOY","GIR","OCH","COU"}

def clean_utterance(text):
    """Remove CHAT codes from the main tier."""
    text = re.sub(r"\x15[\d_]+\x15", "", text)          # timing marks
    text = re.sub(r"&=[^\s]+", "", text)                # gesture/action
    text = re.sub(r"\[//\]|\[/\]|\[\?\]", "", text)     # repair markers
    text = re.sub(r"\[:\s[^\]]*\]", "", text)           # [: repl]
    text = re.sub(r"\[\*[^\]]*\]", "", text)            # [* err]
    text = re.sub(r"\[=\s?[^\]]*\]", "", text)          # [= gloss]
    text = re.sub(r"<|>", "", text)                      # angle brackets
    text = re.sub(r"[‡]", " ", text)                    # tier separator
    text = re.sub(r"\+\.\.\.|\+/\.|\+<|\+,", "", text)   # trailing ops
    text = re.sub(r"&~[^\s]+", "", text)                 # non-word tokens
    text = re.sub(r"&[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_age_months(age_str):
    if not age_str: return None
    m = re.match(r"(\d+);(\d+)?", age_str)
    if not m: return None
    y = int(m.group(1))
    mo = int(m.group(2)) if m.group(2) else 0
    return y*12 + mo

def parse_cha_file(path, corpus_name):
    """Yield one dict per adult utterance."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return
    chi_age = None
    m = re.search(r"\|CHI\|(\d+;[\d.]*)\|", text)
    if m: chi_age = parse_age_months(m.group(1))
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\*(\w+):\s*(.*)", line)
        if m:
            speaker = m.group(1).upper()
            utter = m.group(2)
            j = i + 1
            mor = None
            while j < len(lines) and (lines[j].startswith("\t") or lines[j].startswith("%")):
                if lines[j].startswith("\t"):
                    utter += " " + lines[j].strip()
                elif lines[j].startswith("%mor:"):
                    mor = lines[j][len("%mor:"):].strip()
                j += 1
            i = j
            if speaker in NON_ADULT: continue
            yield {
                "corpus": corpus_name,
                "file": path.name,
                "speaker": speaker,
                "chi_age_months": chi_age,
                "utterance": clean_utterance(utter),
                "mor": mor,
            }
        else:
            i += 1

out = Path("/home/claude/analysis/adult_utterances_all.csv")
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["corpus","file","speaker","chi_age_months","utterance","mor"])
    w.writeheader()
    totals = {}
    for corpus in CORPORA:
        n = 0
        root = ROOT / corpus
        for cha in sorted(root.rglob("*.cha")):
            for u in parse_cha_file(cha, corpus):
                if u["utterance"]:
                    w.writerow(u); n += 1
        totals[corpus] = n
print("Adult utterances per corpus:")
for k,v in totals.items():
    print(f"  {k:20s}: {v:>7,}")
print(f"  {'TOTAL':20s}: {sum(totals.values()):>7,}")
