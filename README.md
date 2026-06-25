# DREAMT Wearable Sleep Staging

## Project Overview

This repository contains an in-progress, reproducible Python workflow for wearable-based sleep stage classification using the DREAMT dataset. The current target is three-class sleep staging: `Wake`, `Non-REM`, and `REM`.

It now includes reusable source modules, staged notebooks, automated tests, engineered feature baselines, PyTorch dataset utilities, 1D CNN training workflows, temporal-context CNN comparisons, CNN-GRU sequence-model utilities, and a multiscale raw/engineered-feature fusion CNN. Raw data, trained checkpoints, and final held-out test results are not committed.

## Current Status

| Stage | Status | Main files |
| --- | --- | --- |
| 1. Dataset inventory | Implemented | `notebooks/01_dataset_overview.ipynb`, `src/data.py` |
| 2. Label mapping audit | Implemented | `notebooks/01_dataset_overview.ipynb`, `src/preprocessing.py` |
| 3. Participant split | Implemented | `src/data.py` |
| 4. Epoch index construction | Implemented | `src/data.py`, `src/preprocessing.py` |
| 5. Training-set EDA | Implemented | `notebooks/02_train_set_eda.ipynb`, `src/plots.py` |
| 6. Engineered feature baselines | Implemented | `notebooks/03_feature_baselines.ipynb`, `src/features.py`, `src/baselines.py`, `src/evaluate.py` |
| 7. PyTorch tensor datasets | Implemented | `src/data.py`, `notebooks/04_cnn_training.ipynb` |
| 8. First 1D CNN training loop | Implemented | `src/models.py`, `src/train.py`, `notebooks/04_cnn_training.ipynb` |
| 9. Single-epoch CNN training choices | Implemented as guarded validation runs | `src/train.py`, `notebooks/04_cnn_training.ipynb` |
| 10. Temporal-context CNN comparison | Implemented as guarded validation runs | `src/train.py`, `notebooks/04_cnn_training.ipynb` |
| 11. Many-to-one CNN-GRU comparison | Implemented as guarded validation runs | `src/models.py`, `src/train.py`, `notebooks/04_cnn_training.ipynb` |
| 12. Many-to-many CNN-GRU aggregation | Implemented as guarded validation runs | `src/models.py`, `src/train.py`, `notebooks/04_cnn_training.ipynb` |
| 13. Validation error analysis | Implemented as guarded validation diagnostics | `src/error_analysis.py`, `notebooks/05_error_analysis.ipynb` |
| 14. Multiscale residual feature-fusion CNN | Implemented as one guarded validation run | `src/data.py`, `src/models.py`, `src/train.py`, `notebooks/04_cnn_training.ipynb` |

The guarded training cells are disabled by default so routine notebook
execution does not launch long experiments. When enabled locally, they write
stage-specific artifacts under `results/stage*/`.

## Dataset

The project uses the DREAMT dataset, which contains wearable physiological signals relevant to sleep analysis. Raw DREAMT data should not be committed to GitHub. Real dataset files should remain local and should be handled according to the dataset access terms and privacy requirements.

Local data files should be placed under `data/raw/`, while intermediate and processed modeling files should remain in local-only data folders. `data/README.md` documents the expected local data products in more detail.

## Prediction Task

The target task is three-class sleep staging:

- `Wake`
- `Non-REM`
- `REM`

Participant-level train, validation, and test splits are used to reduce data leakage risk. The workflow avoids using validation or test information during exploratory analysis, feature engineering, preprocessing decisions, and model selection.

## Participant Split

The project uses a fixed participant-level split so temporally correlated rows
or epochs from the same participant never appear in more than one modeling set.
The default split is 70 participants for training, 15 for validation, and 15 for
testing, created with `random_state=42`.

Reusable split utilities live in `src/data.py`. The saved assignment file is:

- `data/interim/split_assignments.csv`

Downstream preprocessing, EDA, feature extraction, model training, and
evaluation should load this file rather than creating a fresh split. Validation
or test performance should not be used to revise the split.

## Label Mapping

Reusable label-standardization utilities live in `src/preprocessing.py`. The
primary target mapping is:

- `W` and Wake variants -> `Wake`
- `N1`, `N2`, and `N3` -> `Non-REM`
- `R` and REM variants -> `REM`

Downloaded DREAMT `data_64Hz` files may include `P`, which denotes preparation
before PSG recording starts. The primary analysis excludes `P` because it is not
a PSG-scored Wake epoch in the same sense as `W`; treating it as Wake could add
setup-period physiology and motion artifacts to the Wake class. A secondary
sensitivity-analysis option is available with `p_as_wake=True`, matching the
DREAMT `data_100Hz` convention where `P` is treated as Wake. Missing, unknown,
artifact, movement, unscored, and ambiguous labels are excluded rather than
silently forced into one of the three target classes.

## Epoch Preprocessing

Stage 4 uses one PSG sleep epoch as the modeling unit. Reusable epoch utilities
live in `src/preprocessing.py`, and the split-aware index builder lives in
`src/data.py`. DREAMT participant CSVs are assumed to use a 64 Hz grid, so each
30-second epoch should contain `30 * 64 = 1920` rows. Row ranges in the epoch
index are half-open: `start_row` is inclusive and `end_row` is exclusive. Before
segmenting each participant, Stage 4 infers a participant-specific PSG epoch
offset from `Sleep_Stage` transition rows so files that start before the first
30-second scoring boundary are not split on the wrong row alignment.

Each epoch is assigned exactly one mapped target label from `Sleep_Stage` when
all rows in the epoch map consistently to `Wake`, `Non-REM`, or `REM`. Epochs
are excluded when labels are missing, unknown, ambiguous, or change within the
epoch. Epochs are also excluded when the row count differs from the expected
epoch size, when `TIMESTAMP` shows an obvious discontinuity, or when any known
required wearable signal has more than 20% missingness by default. `IBI`
missingness is retained in `missingness_IBI` for later feature handling but does
not by itself exclude an otherwise valid epoch at this stage.

Epoch construction uses `data/interim/split_assignments.csv`, which is a
participant-level split. The builder refuses to create epoch rows for a raw
participant file that is missing from the split assignments, preventing silent
train/validation/test leakage. Scaling and normalization are not fit at this
stage; any later scaler must be fit using training participants only.

## Training-Set EDA

Stage 5 is implemented in `notebooks/02_train_set_eda.ipynb`. The notebook uses
only participants assigned to the `train` split and only valid Stage 4 epochs
for predictive EDA. It loads `data/interim/split_assignments.csv` and
`data/interim/epoch_index.csv`, filters to training epochs, and keeps
validation/test epochs unused.

The EDA covers:

- training class balance and metric implications
- participant-level class distributions
- missingness by signal, participant, and sleep stage
- simple raw-signal epoch summaries by sleep stage
- representative raw multichannel epochs
- within-participant sleep-stage transition matrices
- optional hypnogram-like plots for a few training participants

Reusable EDA and plotting helpers live in `src/plots.py`. Key figures are saved
under `results/figures/` when the notebook is run with local DREAMT artifacts.
Raw signal summaries and representative raw epoch plots require local
participant CSVs under `data/raw/`; the notebook skips those cells cleanly when
raw data are unavailable.

## Engineered Feature Baselines

Stage 6 is implemented in `notebooks/03_feature_baselines.ipynb`, with reusable
feature extraction in `src/features.py` and shared metric helpers in
`src/evaluate.py`. The notebook builds compact epoch-level feature tables for
traditional ML baselines using simple signal summaries for `BVP`, `ACC_X`,
`ACC_Y`, `ACC_Z`, `TEMP`, `EDA`, `HR`, `IBI`, and derived `ACC_MAG`.

Feature rows retain `participant_id`, `epoch_id`, `split`, and `label` so split
integrity and group-level evaluation remain auditable. When local DREAMT files
are available, the notebook writes:

- `data/processed/features_train.csv`
- `data/processed/features_val.csv`
- `data/processed/features_test.csv`

The test feature table is saved for later final evaluation only. Stage 6 model
selection uses participant-level cross-validation within the training split.
The validation split is used for interim evaluation, feature-level permutation
importance, and feature-group permutation importance; the test split is not
used for tuning, feature selection,
permutation importance, or interim conclusions.

Current baseline models in the notebook are:

- majority-class classifier as a sanity check
- balanced elastic-net multinomial logistic regression in a scikit-learn
  imputation/scaling/model pipeline
- balanced XGBoost trained on all engineered features with modest training-CV
  tuning

## Deep Learning Tensor Preparation

Stage 7 adds PyTorch-ready dataset utilities in `src/data.py` and an initial
smoke workflow in `notebooks/04_cnn_training.ipynb`. The basic
`DreamtEpochDataset` returns tensors shaped `(channels, timepoints)` for 1D CNN
training. `DreamtContextDataset` creates neighboring-epoch context windows
without crossing participant boundaries, and `DreamtSequenceDataset` creates
CNN-GRU-style inputs shaped `(sequence_length, channels, timepoints)` with both
many-to-one and many-to-many label support.

Preprocessing metadata is fit from training participants only using mean
imputation and per-channel standardization, then saved locally at:

- `data/processed/preprocessing_metadata.json`

Stage 14 separately fits chunked, training-only mean imputation and
standardization for the engineered feature columns, saved at:

- `data/processed/feature_preprocessing_metadata.json`

For repeated CNN training, raw participant CSVs can be converted once into
per-participant `.npy` arrays under `data/processed/deep/participants/`.
Stage 9 through Stage 12 and Stage 14 notebook configs opt into this processed cache with
`participant_array_cache_dir`; the first configured run builds the cache if its
manifest is missing, and later runs memory-map the arrays instead of reparsing
CSV files.

Validation datasets apply the saved training metadata. The held-out test split
remains unused for model prediction and performance reporting until the final
project comparison.

## First 1D CNN Training Loop

Stage 8 adds a small, reusable single-epoch 1D CNN training loop in
`src/models.py` and `src/train.py`. The first CNN is intentionally modest: its
purpose is to prove that the deep learning plumbing is trustworthy before
larger model variants are explored. The training code supports CPU and GPU
execution through `device="auto"`, converts input batches to `float32` for
PyTorch model training, and monitors validation diagnostics after each epoch.
Full train-set diagnostics are configurable with `train_eval_interval`: `1`
keeps the original every-epoch behavior, while `None` evaluates the train split
only on the final scheduled epoch or the epoch that triggers early stopping.
`train_history.csv` records the per-epoch training objective loss separately
from optional full-train evaluation loss. `training_curves.png` plots the
per-epoch training objective loss and validation macro F1 only. The history file
also records per-epoch phase timings and participant-cache load counts for the
training pass, optional train evaluation pass, and validation pass.

Stage 8 evaluates only the validation split. The held-out test split must not
be used for model prediction or performance reporting until the final project
comparison.

The Stage 8 workflow saves local artifacts under:

- `results/stage8_single_epoch_cnn/train_history.csv`
- `results/stage8_single_epoch_cnn/validation_metrics.csv`
- `results/stage8_single_epoch_cnn/validation_confusion_matrix.csv`
- `results/stage8_single_epoch_cnn/validation_confusion_matrix.png`
- `results/stage8_single_epoch_cnn/training_curves.png`
- `results/stage8_single_epoch_cnn/tiny_overfit_history.csv`
- `results/stage8_single_epoch_cnn/tiny_overfit_curves.png`
- `results/stage8_single_epoch_cnn/checkpoints/best.pt`
- `results/stage8_single_epoch_cnn/checkpoints/last.pt`

## Basic Deep-Learning Training Choices

Stage 9 extends the single-epoch 1D CNN workflow in
`notebooks/04_cnn_training.ipynb` to compare controlled training choices while
keeping the model family fixed. Reusable tuning utilities live in `src/train.py`.
The default screening grid varies train-only class-weighted loss, learning
rate, dropout including no dropout, and weight decay. Validation macro F1 is
the primary model-selection metric, with balanced accuracy, accuracy,
per-class precision/recall/F1, validation loss, and confusion matrices used as
supporting diagnostics.
The Stage 9 notebook sets `train_eval_interval=None` by default so grid searches
continue to validate every epoch without rereading the full train split every
epoch. The training curves therefore show the training objective and validation
macro F1 every epoch, while full train-set evaluation metrics remain available
in the CSV files only where that evaluation pass ran.

Stage 9 still evaluates only the validation split. The held-out test split must
not be used for model prediction, model selection, error analysis, or
performance reporting until the final project comparison.

When run locally, Stage 9 writes artifacts under:

- `results/stage9_training_choices/experiment_summary.csv`
- `results/stage9_training_choices/all_history.csv`
- `results/stage9_training_choices/best_config.json`
- `results/stage9_training_choices/best_validation_confusion_matrix.csv`
- `results/stage9_training_choices/runs/<experiment_id>/train_history.csv`
- `results/stage9_training_choices/runs/<experiment_id>/validation_metrics.csv`
- `results/stage9_training_choices/runs/<experiment_id>/validation_confusion_matrix.csv`
- `results/stage9_training_choices/runs/<experiment_id>/checkpoints/best.pt`
- `results/stage9_training_choices/runs/<experiment_id>/checkpoints/last.pt`

## Temporal-Context CNN Comparison

Stage 10 asks whether neighboring sleep epochs improve validation performance
relative to the simple 1D CNN. It keeps the architecture and training loop
conservative, reusing the same `SleepStageCNN` and `src.train` training
utilities while replacing single-epoch tensors with `DreamtContextDataset`
windows for context models.

To keep the comparison fair, each simple CNN baseline is trained and evaluated
on the same context-eligible center epochs as its paired context CNN. Stage 10
therefore compares the center epoch alone against the same center epoch plus
neighboring context, rather than comparing different validation epoch sets.
The default comparison runs:

- simple 1D CNN on radius-1 context-eligible centers
- context CNN with `context_radius=1`
- simple 1D CNN on radius-2 context-eligible centers
- context CNN with `context_radius=2`

Validation macro F1 remains the primary selection metric, with balanced
accuracy, validation loss, class-level precision/recall/F1, and confusion
matrices used as supporting diagnostics. Stage 10 still evaluates only the
validation split; the held-out test split remains unused.

When run locally, Stage 10 writes artifacts under:

- `results/stage10_temporal_context_cnn/experiment_summary.csv`
- `results/stage10_temporal_context_cnn/all_history.csv`
- `results/stage10_temporal_context_cnn/best_by_context_radius.csv`
- `results/stage10_temporal_context_cnn/best_config.json`
- `results/stage10_temporal_context_cnn/best_validation_confusion_matrix.csv`
- `results/stage10_temporal_context_cnn/runs/<experiment_id>/train_history.csv`
- `results/stage10_temporal_context_cnn/runs/<experiment_id>/validation_metrics.csv`
- `results/stage10_temporal_context_cnn/runs/<experiment_id>/validation_confusion_matrix.csv`
- `results/stage10_temporal_context_cnn/runs/<experiment_id>/checkpoints/best.pt`
- `results/stage10_temporal_context_cnn/runs/<experiment_id>/checkpoints/last.pt`

## CNN-GRU Sequence Comparisons

Stage 11 and Stage 12 extend the deep-learning workflow from concatenated
context windows to explicit epoch sequences. `SleepStageCNNGRU` encodes each
epoch with the CNN trunk, passes epoch embeddings through a GRU, and predicts
sleep-stage labels from the sequence representation.

Stage 11 trains many-to-one CNN-GRU models that predict the center epoch label.
Its follow-up workflow can diagnose an inverse-frequency-weighted checkpoint by
applying powered training priors to saved validation probabilities without
retraining. It also supports a controlled comparison between unweighted
cross-entropy and square-root inverse-frequency weighting through
`class_weight_power`. Stage 12 trains many-to-many CNN-GRU models that predict a
label for every epoch in each input sequence. Because overlapping sequences can
produce multiple probability predictions for the same sleep epoch, Stage 12
aggregates probabilities back to one prediction per epoch.

These stages are available through guarded cells in
`notebooks/04_cnn_training.ipynb` and reusable functions in `src/train.py`.
They continue the same validation-only development pattern: the held-out test
split remains unused until a final comparison is ready.

When run locally, these stages write artifacts under:

- `results/stage11_cnn_gru/experiment_summary.csv`
- `results/stage11_cnn_gru/all_history.csv`
- `results/stage11_cnn_gru/best_by_sequence_length.csv`
- `results/stage11_cnn_gru/best_config.json`
- `results/stage11_cnn_gru/best_validation_confusion_matrix.csv`
- `results/stage11_cnn_gru/prior_correction_summary.csv`
- `results/stage11_cnn_gru_loss_comparison/experiment_summary.csv`
- `results/stage12_cnn_gru_many_to_many/experiment_summary.csv`
- `results/stage12_cnn_gru_many_to_many/all_history.csv`
- `results/stage12_cnn_gru_many_to_many/best_by_sequence_length.csv`
- `results/stage12_cnn_gru_many_to_many/best_config.json`
- `results/stage12_cnn_gru_many_to_many/best_validation_confusion_matrix.csv`

## Multiscale Residual Feature-Fusion CNN

Stage 14 returns to single-epoch prediction with a stronger representation.
`DreamtFeatureFusionDataset` aligns each raw epoch with its Stage 6 engineered
feature row by `(participant_id, epoch_id)`, rejects split or label
disagreement, and exposes the raw epoch index so participant-block sampling
continues to work.

Engineered preprocessing is fit from `features_train.csv` only. It computes
column means and standard deviations in CSV chunks, uses mean imputation, and
stores the transformed matrix as `float32` to avoid the memory-heavy median
path. Validation rows reuse the saved training metadata.

`MultiscaleResidualFusionCNN` uses three raw-signal convolution branches,
GroupNorm residual blocks, and adaptive pooling to 12 temporal bins rather than
collapsing an epoch immediately to one value per channel. A compact MLP encodes
the 72 engineered features before fusion. The fixed configuration uses
unweighted cross-entropy with `label_smoothing=0.05`, AdamW at `3e-4`, gradient
clipping at `1.0`, up to 25 epochs, patience 5, and validation macro F1 for
checkpoint selection. Its notebook launch flag is disabled by default.

After the unweighted run, the notebook also provides one controlled follow-up
that changes only the loss to square-root inverse-frequency weighting
(`class_weighting=True`, `class_weight_power=0.5`). It keeps the architecture,
label smoothing, optimizer settings, seed, and stopping rules fixed, and writes
to a separate results directory.

When run locally, Stage 14 writes artifacts under:

- `results/stage14_multiscale_fusion_cnn/experiment_summary.csv`
- `results/stage14_multiscale_fusion_cnn/all_history.csv`
- `results/stage14_multiscale_fusion_cnn/best_config.json`
- `results/stage14_multiscale_fusion_cnn/best_validation_confusion_matrix.csv`
- `results/stage14_multiscale_fusion_cnn/runs/<experiment_id>/`
- `results/stage14_multiscale_fusion_cnn_sqrt_weighted/`

## Validation Error Analysis

Stage 13 is implemented in `notebooks/05_error_analysis.ipynb`, with reusable
helpers in `src/error_analysis.py`. It analyzes validation predictions from the
implemented model families without loading or evaluating the held-out test
split. The notebook can optionally rebuild Stage 6 validation prediction tables
from saved train/validation feature CSVs, and it discovers prediction artifacts
from completed deep-learning stages.

Stage 13 summarizes native validation metrics, coverage, common error types,
participant-level performance, transition-neighborhood errors, prediction
confidence, high-confidence mistakes, model agreement, and shared-epoch
metrics. Deep-learning runs created after Stage 13 save
`validation_epoch_predictions.csv` for single-output validation models, while
Stage 12 continues to use its aggregated epoch-level prediction outputs.
For older completed deep-learning runs, Stage 13 can export validation
prediction tables from saved `checkpoints/best.pt` files without retraining;
this reruns validation inference only.

When run locally, Stage 13 writes artifacts under:

- `results/stage13_error_analysis/combined_validation_predictions.csv`
- `results/stage13_error_analysis/model_validation_metrics.csv`
- `results/stage13_error_analysis/model_coverage.csv`
- `results/stage13_error_analysis/error_type_summary.csv`
- `results/stage13_error_analysis/participant_error_summary.csv`
- `results/stage13_error_analysis/temporal_error_summary.csv`
- `results/stage13_error_analysis/confidence_summary.csv`
- `results/stage13_error_analysis/confidence_bins.csv`
- `results/stage13_error_analysis/high_confidence_errors.csv`
- `results/stage13_error_analysis/model_disagreement_summary.csv`
- `results/stage13_error_analysis/shared_epoch_model_metrics.csv`
- `results/stage13_error_analysis/figures/confusion_matrix_<model>.png`

## Methods

Implemented methods include:

- Dataset inspection and integrity checks
- Training-set-only exploratory data analysis
- Signal preprocessing and label mapping
- Traditional machine learning baselines using engineered epoch-level features
- PyTorch tensor datasets for single-epoch CNNs, temporal-context CNNs, and
  CNN-GRU sequence models, plus aligned raw/engineered-feature fusion
- PyTorch 1D CNN, CNN-GRU, and multiscale residual fusion models
- Validation monitoring with accuracy, balanced accuracy, macro F1,
  class-specific precision/recall/F1, and confusion matrices
- Validation error analysis across model families, participants, transition
  neighborhoods, prediction confidence, and shared-epoch model agreement

Final held-out test comparisons remain future work. The current modeling
outputs are interim validation diagnostics used to develop and compare model
families without touching the test split.

## Repository Structure

```text
dreamt-wearable-sleep-staging/
  README.md
  pyproject.toml
  data/
    README.md
  notebooks/
    01_dataset_overview.ipynb
    02_train_set_eda.ipynb
    03_feature_baselines.ipynb
    04_cnn_training.ipynb
    05_error_analysis.ipynb
  src/
    baselines.py
    data.py
    evaluate.py
    features.py
    models.py
    plots.py
    preprocessing.py
    train.py
  tests/
  results/
  executive_summary.md
```

The structure is designed to support reproducible data science work:

- `notebooks/` = experimentation and explanation. Notebooks are used for exploration, reporting, and communicating analysis decisions.
- `src/` = reusable project code. Source modules contain importable logic shared across notebooks and scripts.
- `data/` = data documentation, not raw committed data. Real DREAMT files should remain local.
- `tests/` = automated tests for reusable project code.
- `results/` = placeholder files plus local experiment outputs when notebooks or training utilities are run.
- `README.md` = how someone understands and runs the project.

Notebooks are intentionally separated from source code. Notebooks are useful for exploration, explanation, and final reporting, while `src/` holds reusable code that can be imported by notebooks and scripts. This separation improves reproducibility, avoids duplicated logic, and helps reduce data leakage by keeping shared preprocessing, splitting, feature extraction, modeling, and evaluation behavior in one place.

## Setup Instructions

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## How to Run

After placing local DREAMT participant CSVs under `data/raw/`, the expected
workflow is:

1. Place local DREAMT files under `data/raw/`.
2. Run `notebooks/01_dataset_overview.ipynb` to build the participant inventory and label-mapping summaries.
3. Create or load the participant split at `data/interim/split_assignments.csv`.
4. Generate the Stage 4 epoch index at `data/interim/epoch_index.csv`.
5. Run `notebooks/02_train_set_eda.ipynb` for training-set-only EDA.
6. Run `notebooks/03_feature_baselines.ipynb` to build engineered feature CSVs
   and evaluate traditional baselines on the validation split.
7. Run `notebooks/04_cnn_training.ipynb` to build PyTorch datasets, save
   train-only preprocessing metadata, check tensor shapes, run a tiny CNN
   overfit smoke test, and train the first validation-monitored 1D CNN.
8. In the same notebook, enable the guarded Stage 9 cell to compare basic
   single-epoch CNN training choices on the validation split only.
9. In the same notebook, enable the guarded Stage 10 cell to compare simple
   CNN and temporal-context CNN models on matched context-eligible validation
   centers for `context_radius=1` and `context_radius=2`.
10. In the same notebook, enable the guarded Stage 11 and Stage 12 cells to run
    CNN-GRU sequence comparisons on the validation split only.
11. Run `notebooks/05_error_analysis.ipynb` to generate Stage 13 validation
    error-analysis diagnostics from available validation prediction artifacts.
12. In `notebooks/04_cnn_training.ipynb`, enable the guarded Stage 14 cell to
    run the fixed multiscale residual feature-fusion CNN on validation only.
13. Save generated metrics, figures, summaries, and checkpoints under
    stage-specific local `results/` folders.

The dataset overview notebook writes these local intermediate summaries when raw
participant files are available:

- `data/interim/participant_summary.csv`
- `data/interim/label_mapping_summary.csv`
- `data/interim/label_mapping_summary_p_as_wake.csv`
- `data/interim/split_assignments.csv`
- `data/interim/epoch_index.csv`

The feature-baseline notebook writes these local processed feature tables when
the raw files and epoch index are available:

- `data/processed/features_train.csv`
- `data/processed/features_val.csv`
- `data/processed/features_test.csv`

The CNN training notebook and training utilities write train-only preprocessing
metadata when local deep-learning datasets are built:

- `data/processed/preprocessing_metadata.json`
- `data/processed/feature_preprocessing_metadata.json`

Generate the epoch index locally after placing DREAMT participant files under
`data/raw/` and creating `data/interim/split_assignments.csv`:

```python
from src.data import build_epoch_index
from src.preprocessing import summarize_epoch_index

epoch_index = build_epoch_index(
    raw_dir="data/raw",
    split_assignments_path="data/interim/split_assignments.csv",
    output_path="data/interim/epoch_index.csv",
)
summary = summarize_epoch_index(epoch_index)
```

## Results

Final held-out test results are not available yet. The committed files directly
under `results/` are minimal placeholders, not model-performance claims:

- `results/metrics.csv`
- `results/ablation_results.csv`
- `results/confusion_matrix.png`

When local DREAMT files are available, implemented stages can write interim
validation diagnostics and training artifacts such as:

- `results/figures/`
- `results/stage6_feature_baselines/validation_metrics.csv`
- `results/stage6_feature_baselines/validation_confusion_matrix_<model>.csv`
- `results/stage6_feature_baselines/validation_confusion_matrix_<model>.png`
- `results/stage6_feature_baselines/validation_permutation_importance_<model>.csv`
- `results/stage6_feature_baselines/validation_group_permutation_importance_<model>.csv`
- `results/stage6_feature_baselines/feature_group_correlation_matrix.csv`
- `results/stage6_feature_baselines/feature_group_correlation_matrix.png`
- `results/stage8_single_epoch_cnn/`
- `results/stage9_training_choices/experiment_summary.csv`
- `results/stage10_temporal_context_cnn/experiment_summary.csv`
- `results/stage11_cnn_gru/experiment_summary.csv`
- `results/stage12_cnn_gru_many_to_many/experiment_summary.csv`
- `results/stage13_error_analysis/model_validation_metrics.csv`
- `results/stage14_multiscale_fusion_cnn/experiment_summary.csv`

The feature-group correlation matrix reports mean absolute pairwise feature
correlations. Same-group diagonal cells summarize redundancy among distinct
features in the group and exclude each feature's self-correlation.

## Testing

The repository includes automated tests for the reusable code in `src/`,
including participant splitting, label mapping, epoch preprocessing, engineered
features, PyTorch dataset shapes, CNN/CNN-GRU/fusion models, paired-input
training utilities, checkpoint restoration, and staged experiment-summary
writers.

Run the test suite with:

```bash
pytest
```

Run linting with:

```bash
ruff check .
```

## Limitations

This repository does not include raw data, trained models, or final held-out
test results. Stage 6 through Stage 14 report validation diagnostics for model
development, not final benchmark claims. Final comparisons should wait until
model choices are fixed and each selected model class is evaluated once on the
held-out test split.

Known next steps include running the guarded Stage 14 experiment, selecting the
final model families to evaluate, documenting final test-set results, and
continuing to account for participant-level splitting, class imbalance, missing
data, wearable signal quality, and privacy requirements.
