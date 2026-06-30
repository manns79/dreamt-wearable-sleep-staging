# DREAMT Wearable Sleep Staging

A leakage-aware machine learning workflow for classifying polysomnography-derived sleep stage from wearable physiological time series.

## Executive Summary

- **Goal:** Classify 30-second sleep epochs as `Wake`, `Non-REM`, or `REM` using wearable physiological signals. 
- **Data:** [DREAMT](https://physionet.org/content/dreamt/2.2.0/) wearable physiological signals, including blood volume pulse (`BVP`), accelerometry (`ACC`), temperature (`TEMP`), electrodermal activity (`EDA`), heart rate (`HR`), and interbeat interval (`IBI`).
- **Evaluation design:** Fixed participant-level train/validation/test split to reduce leakage from within-participant temporal correlation.
- **Methods:** Naive and traditional ML baselines, CNN variants, sequence models, signal ablations, sensitivity analyses, biologically informed loss-function experiments. 
- **Reproducibility:** Raw DREAMT data are not committed; a small synthetic DREAMT-compatible sample is included for smoke testing the pipeline.
- **Final result:** The best deep learning model meaningfully improves test-set macro F1 over traditional ML and naive baselines: 0.501 vs. 0.435 and 0.266, respectively.

## Key Results

The table below summarizes validation and held-out test performance across the main model families. Macro F1 is reported for validation and test sets; per-class F1 scores are reported on the held-out test set. The best value in each metric column is shown in bold. 

| Model | Validation macro F1 | Test macro F1 | Wake F1 | Non-REM F1 | REM F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority-class baseline | 0.276 | 0.266 | 0.000 | 0.797 | 0.000 |
| Elastic-net logistic regression | 0.375 | 0.408 | 0.487 | 0.575 | **0.162** |
| XGBoost | 0.383 | 0.435 | 0.518 | 0.651 | 0.136 |
| Single-epoch CNN | 0.384 | 0.372 | 0.402 | 0.565 | 0.149 |
| Temporal-context CNN | 0.379 | 0.425 | 0.472 | 0.688 | 0.115 |
| CNN-GRU, many-to-one | 0.284 | 0.329 | 0.418 | 0.409 | 0.161 |
| CNN-GRU, many-to-many | 0.367 | 0.383 | 0.470 | 0.582 | 0.097 |
| MSResCNN-MLP | 0.446 | 0.454 | 0.508 | 0.775 | 0.078 |
| 31-epoch MSResCNN-MLP-TCN | 0.492 | 0.498 | 0.538 | **0.808** | 0.148 |
| 61-epoch MSResCNN-MLP-TCN | 0.506 | 0.500 | 0.563 | 0.792 | 0.145 |
| Transition-regularized 61-epoch MSResCNN-MLP-TCN | **0.510** | **0.501** | **0.564** | 0.793 | 0.146 |

Interpretation:

- The best-performing model combines a multiscale residual CNN-MLP fusion encoder (`MSResCNN-MLP`) with a temporal convolutional network (`TCN`) sequence head. The 61-epoch input window provides approximately 30.5 minutes of temporal context, and transition regularization discourages physiologically rare sleep-stage transitions.

- Deep learning models generally improved over traditional baselines, especially when temporal context was added. One notable exception was REM classification, where elastic-net logistic regression achieved the highest REM F1.

- Adding temporal context improved test macro F1, although the gains were modest in some comparisons. This suggests that neighboring sleep epochs provide useful information, but model architecture and class imbalance remain important constraints.

- Signal-family ablations using `MSResCNN-MLP` suggest that cardiovascular signals (`BVP`, `HR`, and `IBI`) contributed the most useful predictive information among the available wearable signal groups.


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
