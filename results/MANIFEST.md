# Results Manifest

This directory contains a small curated subset of generated result artifacts
that support the findings summarized in the project README. Large local run
outputs, checkpoints, caches, feature tables, and per-epoch prediction files
are intentionally not part of the committed evidence set.

## Curated Summary Files

The `results/summary/` directory is the reader-facing index for committed
results:

- `key_results.csv` contains the metrics reported in the README Key Results
  table, with README display names and the original artifact model names.
- `model_name_crosswalk.csv` maps stage/run artifact names to the README model
  names.
- `signal_ablation_summary.csv` summarizes the Stage 17 signal-family ablation
  metrics and deltas.
- `transition_regularization_summary.csv` summarizes the Stage 19 validation
  sweep over transition-regularization strengths.
- `artifact_manifest.csv` lists the curated files that are intentionally
  committed and the README claim each one supports.

## Stage Guide

| Stage | Purpose | README-facing model or claim | Curated support |
| --- | --- | --- | --- |
| Stage 6 | Traditional feature baselines and sanity baseline. | Majority-class baseline, elastic-net logistic regression, XGBoost. | `results/summary/key_results.csv`; `results/summary/model_name_crosswalk.csv` |
| Stage 9 | Single-epoch raw-signal CNN model selection. | Single-epoch CNN. | `results/summary/key_results.csv`; `results/summary/model_name_crosswalk.csv` |
| Stage 10 | CNN model with neighboring-epoch temporal context. | Temporal-context CNN. | `results/summary/key_results.csv`; `results/summary/model_name_crosswalk.csv` |
| Stage 11 | CNN-GRU sequence model with one label per sequence. | CNN-GRU, many-to-one. | `results/summary/key_results.csv`; `results/summary/model_name_crosswalk.csv` |
| Stage 12 | CNN-GRU sequence model with one label per sequence position. | CNN-GRU, many-to-many. | `results/summary/key_results.csv`; `results/summary/model_name_crosswalk.csv` |
| Stage 14 | Multiscale residual CNN raw-signal branch fused with engineered features. | MSResCNN-MLP. | `results/summary/key_results.csv`; `results/summary/model_name_crosswalk.csv` |
| Stage 15 | TCN sequence head over frozen Stage 14 epoch embeddings with 31-epoch windows. | 31-epoch MSResCNN-MLP-TCN. | `results/summary/key_results.csv`; `results/summary/model_name_crosswalk.csv` |
| Stage 16 | TCN sequence head over frozen Stage 14 epoch embeddings with 61-epoch windows. | 61-epoch MSResCNN-MLP-TCN. | `results/summary/key_results.csv`; `results/summary/model_name_crosswalk.csv` |
| Stage 17 | Signal-family ablations using the multiscale fusion model. | Cardiovascular signals contributed the most useful predictive information. | `results/summary/signal_ablation_summary.csv`; `results/stage17_signal_ablation/figures/stage17_delta_macro_f1.png`; `results/stage17_signal_ablation/figures/stage17_class_f1_deltas.png` |
| Stage 19 | Transition-regularization sweep for the 61-epoch TCN model. | Transition-regularized 61-epoch MSResCNN-MLP-TCN. | `results/summary/transition_regularization_summary.csv`; `results/summary/key_results.csv` |
| Stage 20 | Frozen final held-out test evaluation of validation-selected candidates. | README Key Results table. | `results/summary/key_results.csv`; `results/stage20_final_test_evaluation/figures/validation_test_macro_f1_comparison.png`; `results/stage20_final_test_evaluation/figures/test_per_class_f1.png` |

## Excluded Local Outputs

The local `results/` tree may contain many additional generated files from
notebook runs, including run directories, participant array caches, feature
tables, checkpoints, and per-epoch prediction exports. Those files are useful
for local analysis but are too large or too low-level for the committed
repository. They should be regenerated from the notebooks when needed.
