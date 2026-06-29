# DREAMT Wearable Sleep Staging

A leakage-aware machine learning workflow for classifying PSG-derived sleep stage from wearable physiological time series.

## Overview

This project studies wearable-based sleep staging using the DREAMT dataset. The goal is to predict a three-class PSG sleep-stage label, `Wake`, `Non-REM`, or `REM`, from physiological signals recorded by a wearable device.

The project combines traditional feature-based baselines with deep learning models for raw time-series inputs. It emphasizes reproducible preprocessing, participant-level evaluation, and honest handling of final held-out test results.

## Executive Summary

- **Goal:** Classify 30-second sleep epochs from wearable signals into `Wake`, `Non-REM`, or `REM`.
- **Data:** DREAMT wearable physiological signals, including BVP, accelerometry, temperature, EDA, heart rate, and IBI.
- **Evaluation design:** Fixed participant-level train/validation/test split to reduce leakage from temporally correlated epochs from the same participant.
- **Modeling:** Majority-class baseline, elastic-net logistic regression, XGBoost, 1D CNNs, CNN-GRU sequence models, multiscale fusion CNNs, and frozen-embedding temporal convolutional models.
- **Reproducibility:** Raw DREAMT data are not committed; a small synthetic DREAMT-compatible sample is included for smoke testing the pipeline.
- **Status:** Validation-stage modeling and analysis are implemented. Final held-out test evaluation remains guarded and should be run only after model selection is frozen.

## Key Results

Final held-out test results are not committed in this repository. The table below summarizes the intended comparison format while avoiding unsupported performance claims.

| Model family | Validation macro F1 | Test macro F1 | Notes |
| --- | ---: | ---: | --- |
| Majority-class baseline | N/A | N/A | Sanity-check baseline |
| Elastic-net logistic regression | N/A | N/A | Engineered epoch-level features |
| XGBoost | N/A | N/A | Engineered epoch-level features |
| 1D CNN / temporal-context CNN | N/A | N/A | Raw wearable time-series models |
| CNN-GRU sequence models | N/A | N/A | Explicit temporal sequence modeling |
| Multiscale fusion CNN / frozen-embedding TCN | N/A | N/A | Combines raw signals, engineered features, and longer temporal context |

Interpretation:

- The repository is structured for a validation-first workflow; committed files do not include final held-out test metrics.
- Model-selection logic, validation diagnostics, and final-test scaffolding are implemented, but final benchmark claims should be added only after the guarded final evaluation is run once.
- The synthetic sample verifies that preprocessing and dataset-loading code paths are executable without access to DREAMT.

## Why This Project Matters

This project demonstrates practical data science and machine learning engineering skills on a realistic biomedical time-series problem:

- wearable physiological signal processing at 64 Hz
- participant-level splitting to control leakage
- class-imbalance-aware evaluation with macro F1 and per-class metrics
- reproducible feature extraction and train-only preprocessing
- comparison of interpretable baselines and neural sequence models
- PyTorch dataset utilities for raw epochs, context windows, and sequences
- validation error analysis, signal-family ablation, label-mapping sensitivity, and transition-regularization experiments

## Dataset and Prediction Task

The DREAMT dataset contains wearable physiological signals relevant to sleep analysis. This project uses signals including:

- `BVP`
- `ACC_X`, `ACC_Y`, `ACC_Z`
- `TEMP`
- `EDA`
- `HR`
- `IBI`

Raw DREAMT files are not committed to GitHub and should remain local according to dataset access terms and privacy requirements. Local real-data files are expected under `data/raw/`.

For reviewers without DREAMT access, the repository includes a small committed synthetic sample under `data/synthetic/`. These files are not real participant records. They use the same CSV schema expected by the project loaders and contain four synthetic participants with 31 valid 30-second epochs each. The sample is intended for pipeline smoke tests, not model-performance claims.

The target is a three-class sleep-stage label:

- `Wake`
- `Non-REM`
- `REM`

PSG labels `N1`, `N2`, and `N3` are mapped to `Non-REM`. DREAMT `data_64Hz` files may include `P`, a preparation-stage label before PSG recording starts. The primary workflow excludes `P`; treating `P` as `Wake` is implemented only as a documented sensitivity analysis.

## Evaluation Design

The project uses a fixed participant-level split: participants, not individual epochs, are assigned to train, validation, or test sets. This matters because nearby sleep epochs from the same participant are temporally correlated; splitting epochs independently would overstate generalization.

Evaluation principles:

- preprocessing and normalization statistics are fit on training participants only
- model selection and explanatory analyses use validation data only
- final held-out test evaluation is guarded and should run once after model selection is frozen
- macro F1 is the primary metric because the target classes are imbalanced

Supporting metrics include per-class precision/recall/F1, balanced accuracy, confusion matrices, participant-level macro F1, total sleep time error, and REM duration error.

## Modeling Approach

The long development workflow is organized around a few methodological groups:

**Traditional baselines**

- majority-class classifier
- elastic-net multinomial logistic regression
- XGBoost on engineered epoch-level signal summaries

**Deep learning models**

- single-epoch 1D CNN
- temporal-context CNN using neighboring epochs
- CNN-GRU sequence models
- multiscale residual fusion CNN combining raw signals and engineered features
- frozen-embedding many-to-many temporal convolutional model

**Explanatory and robustness analyses**

- validation error analysis across model families
- signal-family ablation
- `P`-as-`Wake` sensitivity analysis
- train-label transition-regularization ablation

## Repository Structure

```text
dreamt-wearable-sleep-staging/
  README.md
  pyproject.toml
  data/
    README.md
    synthetic/          # committed synthetic smoke-test sample
  notebooks/            # staged analysis and modeling notebooks
  src/                  # reusable preprocessing, data, modeling, and evaluation code
  tests/                # automated tests for reusable code paths
  results/              # placeholders plus local-only generated artifacts
```

The repository separates exploratory notebooks from reusable source code. Raw DREAMT data, trained checkpoints, processed local artifacts, and final test outputs are not committed.

## Reproducibility

Set up the project on Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To run with real DREAMT files, place participant CSVs under `data/raw/`, then run the notebooks in order:

1. `notebooks/01_dataset_overview.ipynb`
2. `notebooks/02_train_set_eda.ipynb`
3. `notebooks/03_feature_baselines.ipynb`
4. `notebooks/04_cnn_training.ipynb`
5. `notebooks/05_error_analysis.ipynb`
6. `notebooks/06_signal_ablation.ipynb`
7. `notebooks/07_p_as_wake_sensitivity.ipynb`
8. `notebooks/08_transition_regularization.ipynb`
9. `notebooks/09_final_test_evaluation.ipynb`

Long-running training cells and final-test cells are guarded or disabled by default. Enable them intentionally after verifying prerequisites.

### Synthetic Data Smoke Run

To smoke-test the pipeline without DREAMT access, copy the committed synthetic CSV files into the ignored local raw-data directory:

```bash
mkdir -p data/raw data/interim
cp data/synthetic/S*_whole_df.csv data/raw/
```

Create the matching local split assignment:

```bash
cat > data/interim/split_assignments.csv <<'EOF'
participant_id,split
S901,train
S902,train
S903,validation
S904,test
EOF
```

The synthetic sample supports preprocessing, feature construction, and dataset-loading smoke tests. It is too small for meaningful model evaluation.

## Testing

Run the test suite:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

The tests cover participant splitting, label mapping, epoch preprocessing, engineered features, PyTorch dataset shapes, model utilities, training helpers, sensitivity-analysis helpers, and compatibility of the committed synthetic data with the raw-data pipeline.

## Limitations and Future Work

Wearable-only sleep staging is challenging because wearable signals are indirect proxies for PSG sleep stage. Minority classes such as `REM` can be especially difficult under class imbalance and participant heterogeneity.

Current limitations:

- final held-out test results are not committed
- local validation artifacts and checkpoints are intentionally excluded
- DREAMT is a finite dataset, so participant-level evaluation and external validation are important
- synthetic data are included only for reproducibility smoke tests, not for training meaningful models

Future work includes final guarded test evaluation, external validation, probability calibration, additional temporal modeling, richer uncertainty analysis, and deployment-oriented improvements for efficient inference and monitoring.
