# DREAMT Wearable Sleep Staging

## Project Overview

This repository contains an in-progress, reproducible Python workflow for wearable-based sleep stage classification using the DREAMT dataset. The current target is three-class sleep staging: `Wake`, `Non-REM`, and `REM`.

It now includes reusable source modules, staged notebooks, automated tests, engineered feature baselines, PyTorch dataset utilities, 1D CNN training workflows, temporal-context CNN comparisons, CNN-GRU sequence-model utilities, a multiscale raw/engineered-feature fusion CNN, a frozen-embedding many-to-many temporal convolutional model, signal-family ablations, P-as-Wake sensitivity, transition regularization, and a guarded final-test evaluation framework. Raw data, trained checkpoints, and final held-out test results are not committed.

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
| 15. Frozen Stage 14 embedding TCN | Implemented as one guarded many-to-many validation run | `src/data.py`, `src/models.py`, `src/train.py`, `notebooks/04_cnn_training.ipynb` |
| 16. 61-epoch frozen-embedding TCN | Implemented as guarded validation and seed-replication runs | `src/data.py`, `src/models.py`, `src/train.py`, `notebooks/04_cnn_training.ipynb` |
| 17. Signal-family ablation | Implemented as guarded validation-only explanatory runs | `src/signal_ablation.py`, `notebooks/06_signal_ablation.ipynb` |
| 18. P-as-Wake sensitivity | Implemented as one guarded validation-only sensitivity run | `src/p_as_wake_sensitivity.py`, `notebooks/07_p_as_wake_sensitivity.ipynb` |
| 19. Transition regularization | Implemented as guarded validation-only lambda/seed ablation | `src/transition_regularization.py`, `notebooks/08_transition_regularization.ipynb` |
| 20. Final held-out test evaluation | Framework prepared; final test run remains guarded | `src/final_evaluation.py`, `notebooks/09_final_test_evaluation.ipynb` |

The guarded training cells are disabled by default so routine notebook
execution does not launch long experiments. When enabled locally, they write
stage-specific artifacts under `results/stage*/`.

## Dataset

The project uses the DREAMT dataset, which contains wearable physiological signals relevant to sleep analysis. Raw DREAMT data should not be committed to GitHub. Real dataset files should remain local and should be handled according to the dataset access terms and privacy requirements.

Local data files should be placed under `data/raw/`, while intermediate and processed modeling files should remain in local-only data folders. `data/README.md` documents the expected local data products in more detail.

For boot-camp review and reproducibility without DREAMT access, this repository
includes a small synthetic DREAMT-compatible sample under `data/synthetic/`.
These files are not real participant records. They use the expected 64 Hz CSV
schema and contain four synthetic participants, each with 31 valid 30-second
epochs, so the preprocessing, feature extraction, and PyTorch dataset utilities
can be smoke-tested after copying them into a local `data/raw/` directory.

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

Stage 18 implements the P-as-Wake option as a sensitivity analysis. It should
not replace the primary drop-P analysis unless that decision is made before any
held-out test evaluation.

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
For Stage 18, the same builder can be run with `p_as_wake=True` and restricted
to train/validation splits so the sensitivity analysis does not create held-out
test prediction artifacts.

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

## Frozen-Embedding Many-To-Many TCN

Stage 15 reuses the best square-root-weighted Stage 14 checkpoint as a frozen
epoch encoder. It caches the 160-dimensional concatenated raw and engineered
embedding for every training and validation epoch, then trains a compact
non-causal temporal convolutional network over consecutive 31-epoch windows.
The encoder is not updated during this first temporal experiment.

The TCN uses four residual dilation levels `(1, 2, 4, 8)` and predicts one
sleep-stage label at every sequence position. Overlapping windows use
inverse-epoch-coverage loss weights, preventing central epochs from contributing
more total training loss merely because they occur in more windows. Validation
probabilities are aggregated back to one prediction per epoch. Center-weighted
aggregation is the primary checkpoint metric, with uniform aggregation saved as
a secondary diagnostic.

The first run creates a reusable local embedding cache under:

- `data/processed/stage15_embeddings/train_embeddings.npy`
- `data/processed/stage15_embeddings/train_epoch_index.csv`
- `data/processed/stage15_embeddings/validation_embeddings.npy`
- `data/processed/stage15_embeddings/validation_epoch_index.csv`
- `data/processed/stage15_embeddings/manifest.json`

Stage 15 result artifacts are written under:

- `results/stage15_temporal_fusion_tcn/experiment_summary.csv`
- `results/stage15_temporal_fusion_tcn/all_history.csv`
- `results/stage15_temporal_fusion_tcn/best_config.json`
- `results/stage15_temporal_fusion_tcn/best_validation_confusion_matrix.csv`
- `results/stage15_temporal_fusion_tcn/runs/<experiment_id>/`

The Stage 15 replication workflow preserves the completed seed-42 run, trains
the identical configuration with seeds 43 and 44, and creates an equal-weight
probability ensemble across all three seeds. It reports every seed rather than
selecting the best one, plus the across-seed mean and sample standard deviation.
Replication and ensemble artifacts are written under:

- `results/stage15_temporal_fusion_tcn_seed_replication/seed_43/`
- `results/stage15_temporal_fusion_tcn_seed_replication/seed_44/`
- `results/stage15_temporal_fusion_tcn_seed_replication/seed_member_metrics.csv`
- `results/stage15_temporal_fusion_tcn_seed_replication/seed_metric_statistics.csv`
- `results/stage15_temporal_fusion_tcn_seed_replication/ensemble_validation_metrics.csv`
- `results/stage15_temporal_fusion_tcn_seed_replication/ensemble_validation_epoch_predictions.csv`
- `results/stage15_temporal_fusion_tcn_seed_replication/ensemble_validation_confusion_matrix.csv`
- `results/stage15_temporal_fusion_tcn_seed_replication/seed_ensemble_summary.csv`

## 61-Epoch-Window Temporal Follow-Up

Stage 16 repeats the Stage 15 frozen-embedding many-to-many TCN with one
controlled change: the sequence length increases from 31 to 61 epochs. The
longer window matches the existing TCN's theoretical receptive field. The
frozen Stage 14 encoder, reusable Stage 15 embedding cache, TCN architecture,
loss and aggregation choices, optimizer settings, 30-epoch training limit,
early stopping, and square-root class weighting remain fixed.

The workflow first trains and reviews the seed-42 model. After that run exists,
a separate guarded cell trains seeds 43 and 44 and creates an equal-weight
probability ensemble across seeds 42, 43, and 44 without retraining seed 42.

Stage 16 artifacts are written under:

- `results/stage16_temporal_fusion_tcn_s61/experiment_summary.csv`
- `results/stage16_temporal_fusion_tcn_s61/all_history.csv`
- `results/stage16_temporal_fusion_tcn_s61/best_config.json`
- `results/stage16_temporal_fusion_tcn_s61/runs/<experiment_id>/`
- `results/stage16_temporal_fusion_tcn_s61_seed_replication/seed_43/`
- `results/stage16_temporal_fusion_tcn_s61_seed_replication/seed_44/`
- `results/stage16_temporal_fusion_tcn_s61_seed_replication/seed_member_metrics.csv`
- `results/stage16_temporal_fusion_tcn_s61_seed_replication/seed_metric_statistics.csv`
- `results/stage16_temporal_fusion_tcn_s61_seed_replication/ensemble_validation_metrics.csv`
- `results/stage16_temporal_fusion_tcn_s61_seed_replication/seed_ensemble_summary.csv`

## Signal-Family Ablation

Stage 17 is implemented in `notebooks/06_signal_ablation.ipynb`, with reusable
helpers in `src/signal_ablation.py`. It is a validation-only explanatory
analysis that reuses the fixed Stage 14 square-root-weighted fusion recipe while
ablating raw channels and engineered features by signal family.

When run locally, Stage 17 writes isolated artifacts under:

- `results/stage17_signal_ablation/experiment_summary.csv`
- `results/stage17_signal_ablation/all_history.csv`
- `results/stage17_signal_ablation/best_config.json`
- `results/stage17_signal_ablation/features/<ablation_id>/`
- `results/stage17_signal_ablation/runs/<ablation_id>/`

## P-As-Wake Sensitivity

Stage 18 is implemented in `notebooks/07_p_as_wake_sensitivity.ipynb`, with
reusable helpers in `src/p_as_wake_sensitivity.py`. It rebuilds train and
validation inputs with `P` preparation labels mapped to `Wake`, then retrains
one representative model: the Stage 14 square-root-weighted multiscale residual
feature-fusion CNN. This is a sensitivity analysis for reporting robustness to
the original DREAMT paper convention, not the main result.

Stage 18 intentionally writes only train/validation modeling artifacts and
checks its output directory for held-out test split rows. When run locally, it
writes artifacts under:

- `results/stage18_p_as_wake_sensitivity/epoch_index_p_as_wake_train_validation.csv`
- `results/stage18_p_as_wake_sensitivity/features/features_train.csv`
- `results/stage18_p_as_wake_sensitivity/features/features_val.csv`
- `results/stage18_p_as_wake_sensitivity/experiment_summary.csv`
- `results/stage18_p_as_wake_sensitivity/all_history.csv`
- `results/stage18_p_as_wake_sensitivity/best_config.json`
- `results/stage18_p_as_wake_sensitivity/runs/p_as_wake_stage14_sqrt_weighted/`

## Transition-Regularization Ablation

Stage 19 is implemented in `notebooks/08_transition_regularization.ipynb`, with
reusable helpers in `src/transition_regularization.py`. It estimates a
train-label transition matrix from training epochs only, converts the smoothed
transition probabilities into a zero-diagonal cost matrix, and trains the
Stage 16 61-epoch frozen-embedding TCN with an added adjacent-probability
transition penalty. The existing Stage 16 three-seed ensemble is reused as the
`lambda_transition = 0.0` baseline rather than retrained.

The guarded run trains the nonzero lambda grid `0.001`, `0.01`, `0.05`, `0.1`,
`0.25`, `0.5`, `0.75`, and `1.0` across seeds 42, 43, and 44, then creates one
equal-weight validation ensemble per lambda. Completed seed runs and completed
per-lambda ensemble artifacts are reused when `skip_completed=True`, so expanding
the grid preserves the initial lambda results instead of retraining or rewriting
them. Transition counts are built from full training epoch indexes, not sliding
windows, and transitions are counted only within participants across consecutive
epoch IDs. Validation and held-out test labels are not used to construct the
transition cost matrix.

When run locally, Stage 19 writes artifacts under:

- `results/stage19_transition_regularization/transition_counts_train.csv`
- `results/stage19_transition_regularization/transition_probabilities_train.csv`
- `results/stage19_transition_regularization/transition_cost_matrix.csv`
- `results/stage19_transition_regularization/lambda_0_0/`
- `results/stage19_transition_regularization/lambda_<value>/seed_<seed>/`
- `results/stage19_transition_regularization/lambda_<value>/ensemble_validation_metrics.csv`
- `results/stage19_transition_regularization/lambda_<value>/ensemble_validation_epoch_predictions.csv`
- `results/stage19_transition_regularization/lambda_<value>/validation_predicted_transition_matrix.csv`
- `results/stage19_transition_regularization/lambda_<value>/validation_true_transition_matrix.csv`
- `results/stage19_transition_regularization/experiment_summary.csv`
- `results/stage19_transition_regularization/baseline_comparison.csv`
- `results/stage19_transition_regularization/all_history.csv`
- `results/stage19_transition_regularization/best_config.json`

## Final Held-Out Test Evaluation

Stage 20 is implemented in `notebooks/09_final_test_evaluation.ipynb`, with
reusable helpers in `src/final_evaluation.py`. It is a guarded final-evaluation
framework, not a validation-development stage. The notebook first builds a
candidate registry from validation artifacts only, then allows that registry to
be frozen after Stage 19 completes. The final held-out test evaluation remains
disabled unless `RUN_FINAL_TEST_EVALUATION = True` is set explicitly.

The final candidate list includes the Stage 6 majority-class, logistic
regression, and XGBoost baselines; the validation-selected best models from
Stages 9, 10, 11, 12, and 14; the equal-weight seed ensembles from Stages 15 and
16; and the Stage 19 equal-weight seed ensemble selected by validation macro F1
after Stage 19 finishes.

Stage 20 compares validation and test performance using macro F1 as the primary
metric, per-class precision/recall/F1, confusion matrices, participant-level
macro F1, total sleep time error, and REM duration error. Duration errors are
computed at participant level from 30-second epochs and summarized as signed
and absolute minute errors.

When run locally, Stage 20 writes artifacts under:

- `results/stage20_final_test_evaluation/candidate_registry_draft.csv`
- `results/stage20_final_test_evaluation/final_candidate_registry.csv`
- `results/stage20_final_test_evaluation/final_candidate_registry_manifest.json`
- `results/stage20_final_test_evaluation/validation_metrics.csv`
- `results/stage20_final_test_evaluation/test_metrics.csv`
- `results/stage20_final_test_evaluation/validation_test_metric_comparison.csv`
- `results/stage20_final_test_evaluation/test_participant_macro_f1.csv`
- `results/stage20_final_test_evaluation/test_duration_errors.csv`
- `results/stage20_final_test_evaluation/duration_error_summary.csv`
- `results/stage20_final_test_evaluation/figures/`

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
- Frozen-embedding many-to-many temporal convolution with overlap-aware loss
  weighting, 31- and 61-epoch windows, and epoch-level probability aggregation
- Validation-only signal-family ablations and P-as-Wake label-mapping
  sensitivity analysis
- Train-label transition regularization as a validation-only Stage 16 ensemble
  ablation
- Validation monitoring with accuracy, balanced accuracy, macro F1,
  class-specific precision/recall/F1, and confusion matrices
- Validation error analysis across model families, participants, transition
  neighborhoods, prediction confidence, and shared-epoch model agreement
- Guarded final held-out test framework with validation/test comparisons,
  participant-level macro F1, total sleep time error, REM duration error, and
  presentation-ready summary figures

Final held-out test execution remains guarded until the candidate registry is
frozen. The current modeling outputs are interim validation diagnostics used to
develop and compare model families without touching the test split.

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
    06_signal_ablation.ipynb
    07_p_as_wake_sensitivity.ipynb
    08_transition_regularization.ipynb
    09_final_test_evaluation.ipynb
  src/
    baselines.py
    data.py
    evaluate.py
    final_evaluation.py
    features.py
    models.py
    p_as_wake_sensitivity.py
    plots.py
    preprocessing.py
    signal_ablation.py
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
13. After the weighted Stage 14 checkpoint exists, enable the guarded Stage 15
    cell to cache frozen embeddings and train the many-to-many TCN.
14. Enable the guarded Stage 15 seed-replication cell to train seeds 43 and 44
    and create the equal-weight seed ensemble without retraining seed 42.
15. Enable the guarded Stage 16 cell to train the seed-42 61-epoch-window TCN.
16. After reviewing that run, enable the Stage 16 seed-replication cell to
    train seeds 43 and 44 and create the three-seed equal-weight ensemble.
17. Run `notebooks/06_signal_ablation.ipynb` for Stage 17 validation-only
    signal-family ablation diagnostics.
18. Run `notebooks/07_p_as_wake_sensitivity.ipynb` for the guarded Stage 18
    validation-only P-as-Wake sensitivity analysis.
19. Run `notebooks/08_transition_regularization.ipynb` for the guarded Stage 19
    transition-regularization lambda/seed ablation.
20. Run `notebooks/09_final_test_evaluation.ipynb` in Phase 1 mode to inspect
    and freeze the final candidate registry after Stage 19 completes. Run the
    final test-evaluation cell only once the registry is frozen.
21. Save generated metrics, figures, summaries, and checkpoints under
    stage-specific local `results/` folders.

### Synthetic Data Smoke Run

The committed synthetic sample lives in `data/synthetic/` so it remains clearly
separate from local raw DREAMT data. To run the code against the synthetic
sample, copy those CSVs into your ignored local raw-data folder:

```bash
mkdir -p data/raw data/interim
cp data/synthetic/S*_whole_df.csv data/raw/
```

Then create a small participant split for the synthetic IDs:

```bash
cat > data/interim/split_assignments.csv <<'EOF'
participant_id,split
S901,train
S902,train
S903,validation
S904,test
EOF
```

The default split creator is intended for the full 100-participant DREAMT
cohort, so use the explicit split above for the synthetic sample. After that,
the usual Stage 1 through Stage 7 preprocessing, feature, and dataset utilities
can run against `data/raw/`. The synthetic sample is deliberately small and is
intended for code-path smoke tests rather than meaningful model performance.

The dataset overview notebook writes these local intermediate summaries when raw
participant files are available:

- `data/interim/participant_summary.csv`
- `data/interim/label_mapping_summary.csv`
- `data/interim/label_mapping_summary_p_as_wake.csv`
- `data/interim/split_assignments.csv`
- `data/interim/epoch_index.csv`
- `results/stage18_p_as_wake_sensitivity/epoch_index_p_as_wake_train_validation.csv`

The feature-baseline notebook writes these local processed feature tables when
the raw files and epoch index are available:

- `data/processed/features_train.csv`
- `data/processed/features_val.csv`
- `data/processed/features_test.csv`
- `results/stage18_p_as_wake_sensitivity/features/features_train.csv`
- `results/stage18_p_as_wake_sensitivity/features/features_val.csv`

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
- `results/stage15_temporal_fusion_tcn/experiment_summary.csv`

The feature-group correlation matrix reports mean absolute pairwise feature
correlations. Same-group diagonal cells summarize redundancy among distinct
features in the group and exclude each feature's self-correlation.

## Testing

The repository includes automated tests for the reusable code in `src/`,
including participant splitting, label mapping, epoch preprocessing, engineered
features, PyTorch dataset shapes, CNN/CNN-GRU/fusion models, paired-input
training utilities, checkpoint restoration, signal ablations, P-as-Wake
sensitivity helpers, and staged experiment-summary writers.

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
test results. Stage 6 through Stage 18 report validation diagnostics for model
development and sensitivity analysis, not final benchmark claims. Final
comparisons should wait until model choices are fixed and each selected model
class is evaluated once on the held-out test split.

Known next steps include reviewing the validation-only Stage 17 and Stage 18
diagnostics, selecting the final model/story before any held-out evaluation,
documenting final test-set results after that freeze, and continuing to account
for participant-level splitting, class imbalance, missing data, wearable signal
quality, and privacy requirements.
