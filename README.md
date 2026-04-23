# Supplementary materials

Accompanying "Distributional and lexical signatures of superordinate nouns in Mandarin Chinese: A cross-linguistic test of Choe and Papafragou (2026)."

**Note on scope.** This revised package uses only Xu et al. (2021) for Chinese AoA and Kuperman et al. (2012) for English AoA. All analyses based on Zhang et al. (2024) and Brysbaert & Biemiller (2017) have been removed per revision request.

## Study 1 — Mandarin CHILDES distributional analysis

Pipeline (run in order):
- `13_parse_all_corpora.py` — Parse all 11 CHILDES corpora; produces `adult_utterances_all.csv` (~60 MB; regenerable, not included in this zip).
- `17_target_counts_all.py` — Per-item counts across all 11 corpora; produces `target_counts_all.csv`.
- `14_extract_contexts_all.py` — Extract 6 Mandarin distributional contexts per target item; produces `study1_context_counts_all.csv`.
- `15c_cluster_robust_v2.py` — Cluster-robust logistic regression with items as clusters; produces `study1_cluster_robust.csv`.
- `16_fig_study1_final.py` — Figure 1.

## Study 2 — Bilingual AoA comparison + item-level regressions (REVISED)

Pipeline (run after Study 1):
- `20_study2_xu_kup.py` — Merge Xu (2021) and Kuperman (2012) for item-level bilingual file; coverage + paired comparisons + z-score analyses; produces `bilingual_xu_kup.csv`.
- `21_expanded_contexts.py` — Extract Mandarin distributional cues for the 122-item expanded inventory; produces `expanded_contexts.csv`.
- `22_item_level_models.py` — **Core reviewer analysis.** OLS regressions of AoA on distributional-cue index, log corpus frequency, and semantic level, on the expanded 115-item set. Also runs the Choe-strict subset and the sufficiency-classifier test. Produces `model_cn_M{0..4}.csv`, `classifier_level_from_cues.csv`.
- `23_figures_final_v2.py` — Figures 2–6.

## Figures

- `fig_study1.png` — Figure 1 (Study 1 distributional cues by level, with cluster-robust stars).
- `fig_bilingual_sup.png` — Figure 2 (paired English vs Chinese AoA for Choe's 7 superordinates).
- `fig_bilingual_diff.png` — Figure 3 (per-domain super-vs-basic z-score diff, both languages).
- `fig_bilingual_scatter.png` — Figure 4 (item-level bilingual AoA scatter).
- `fig_distribution_vs_aoa.png` — Figure 5 (NEW: distributional cue index and frequency vs Chinese AoA — the core reviewer-requested analysis).
- `fig_morphology.png` — Figure 6 (morphological length by level, Choe items).

## Data files

Study 1 outputs:
- `target_counts_all.csv` — Per-item counts, all 11 corpora broken out.
- `study1_context_counts_all.csv` — Per-item × per-context co-occurrence counts.
- `study1_chi2_all.csv` — Aggregate chi-square results.
- `study1_cluster_robust.csv` — Cluster-robust logistic regression output (manuscript Table 4).

Study 2 outputs:
- `bilingual_xu_kup.csv` — All items with both Xu (Chinese) and Kuperman (English) AoA, plus Mandarin distributional cue data, character count, frequency.
- `expanded_contexts.csv` — The 122-item expanded inventory with Mandarin distributional cues from CHILDES.
- `choe_items_enriched.csv` — The 21 Choe items with all predictors joined.
- `classifier_level_from_cues.csv` — Sufficiency classifier results (§2.2.3).
- `model_cn_M0.csv` through `model_cn_M4.csv` — Regression sequence (manuscript Table 5).
- `items_enriched.csv`, `further_expanded.csv` — Chinese-only precursor files (used by script 22).

## Reproducibility

Python 3.12 with pandas, numpy, scipy, matplotlib + Noto Sans CJK font for figure rendering.

Data sources (must be placed as indicated):
- 11 Mandarin CHILDES corpora in CHAT format in `/home/claude/corpora/{corpus_name}/`.
- Xu et al. (2021): `/mnt/project/2021Xu.xlsx`.
- Kuperman et al. (2012): `/mnt/project/2012KupermanAoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx`.
