# DREAMT Wearable Sleep Staging

## Project Overview

This repository is the initial scaffold for a deep learning project focused on wearable-based sleep stage classification using the DREAMT dataset. The goal is to develop a reproducible deep learning workflow that uses physiological signals from wearable devices to classify sleep stages.

The repository is currently in an initial scaffold stage. Core implementation details, experiments, model training code, and final results will be added as the project develops.

## Dataset

The project will use the DREAMT dataset, which contains wearable physiological signals relevant to sleep analysis. Raw DREAMT data should not be committed to GitHub. Real dataset files should remain local and should be handled according to the dataset access terms and privacy requirements. 

Local data files should be placed under `data/raw/`, while intermediate and processed modeling files should remain in local-only data folders. A small synthetic dataset will be added later for illustrative purposes, but it will be clearly separated from the real DREAMT data.

## Prediction Task

The target task will likely be three-class sleep staging:

- Wake
- REM
- Non-REM

Participant-level train, validation, and test splits will be used wherever possible to reduce data leakage risk. The project will avoid using validation or test information during exploratory analysis, feature engineering, preprocessing decisions, and model selection.

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
The validation split is used for interim evaluation and XGBoost permutation
importance; the test split is not used for tuning, feature selection,
permutation importance, or interim conclusions.

Current baseline models in the notebook are:

- majority-class classifier as a sanity check
- balanced elastic-net multinomial logistic regression in a scikit-learn
  imputation/scaling/model pipeline
- XGBoost trained on all engineered features with modest training-CV tuning

## Deep Learning Tensor Preparation

Stage 7 adds PyTorch-ready dataset utilities in `src/data.py` and an initial
smoke workflow in `notebooks/04_cnn_training.ipynb`. The basic
`DreamtEpochDataset` returns tensors shaped `(channels, timepoints)` for 1D CNN
training. `DreamtContextDataset` creates neighboring-epoch context windows
without crossing participant boundaries, and `DreamtSequenceDataset` creates
CNN-GRU-style inputs shaped `(sequence_length, channels, timepoints)` with both
many-to-one and many-to-many label support.

Preprocessing metadata is fit from training participants only using median
imputation and per-channel standardization, then saved locally at:

- `data/processed/preprocessing_metadata.json`

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
`train_history.csv` also records per-epoch phase timings and participant-cache
load counts for the training pass, optional train evaluation pass, and
validation pass.

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
epoch.

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
matrices used as supporting diagnostics. Broader hyperparameter tuning is held
for Stage 11, where the context radius and other context-CNN choices can be
varied more systematically. Stage 10 still evaluates only the validation split;
the held-out test split remains unused.

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

## Methods

Implemented and planned methods include:

- Dataset inspection and integrity checks
- Training-set-only exploratory data analysis
- Signal preprocessing and label mapping
- Traditional machine learning baselines using engineered epoch-level features
- PyTorch tensor datasets for single-epoch CNNs, temporal-context CNNs, and
  CNN-GRU sequence models
- PyTorch deep learning models, likely starting with a 1D CNN
- Evaluation using accuracy, balanced accuracy, macro F1, class-specific F1 scores, and confusion matrices

The traditional baseline workflow is now implemented as an interim validation
stage. Deep learning models and final held-out test comparisons remain future
work.

## Repository Structure

```text
dreamt-wearable-sleep-staging/
├── README.md
├── pyproject.toml
├── .gitignore
├── data/
├── notebooks/
├── src/
├── results/
└── executive_summary.md
```

The structure is designed to support reproducible data science work:

- `notebooks/` = experimentation and explanation. Notebooks are used for exploration, reporting, and communicating analysis decisions.
- `src/` = reusable project code. Source modules contain importable logic shared across notebooks and scripts.
- `data/` = data documentation, not raw committed data. Real DREAMT files should remain local.
- `results/` = outputs produced by experiments, such as metrics tables, confusion matrices, and figures.
- `README.md` = how someone understands and runs the project.

Notebooks are intentionally separated from source code. Notebooks are useful for exploration, explanation, and final reporting, while `src/` holds reusable code that can be imported by notebooks and scripts. This separation improves reproducibility, avoids duplicated logic, and helps reduce data leakage by keeping shared preprocessing, splitting, feature extraction, modeling, and evaluation behavior in one place.

## Setup Instructions

Create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```


## How to Run

This project is not fully implemented yet. The expected workflow will be:

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
10. Continue moving reusable logic from notebooks into `src/` as new modeling
   stages mature.
11. Save generated metrics and figures under `results/`.

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
- `data/processed/preprocessing_metadata.json`

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

As implementation develops, this section will include concrete commands for preprocessing, training, evaluation, and report generation.

## Results

Final held-out test results are not available yet. Placeholder files are
included under `results/` to show intended outputs, and Stage 6 can additionally
write interim validation diagnostics:

- `results/metrics.csv`
- `results/ablation_results.csv`
- `results/confusion_matrix.png`
- `results/figures/`
- `results/stage6_validation_metrics.csv`
- `results/stage6_xgboost_validation_permutation_importance.csv`
- `results/stage9_training_choices/experiment_summary.csv`
- `results/stage10_temporal_context_cnn/experiment_summary.csv`

## Limitations

This scaffold does not include raw data, trained models, or final experimental
results. Stage 6 reports validation diagnostics only; final comparisons should
wait until the modeling pipeline is mature and each selected model class is
evaluated once on the held-out test split. Future work must carefully address
participant-level splitting, class imbalance, missing data, wearable signal
quality, and privacy requirements.
