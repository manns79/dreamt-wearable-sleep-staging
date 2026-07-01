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

- The strongest deep learning models improved over traditional baselines, especially when temporal context was added. One notable exception was REM classification, where elastic-net logistic regression achieved the highest REM F1.

- Adding temporal context improved test macro F1, although the gains were modest in some comparisons. This suggests that neighboring sleep epochs provide useful information, but model architecture and class imbalance remain important constraints.

- Signal-family ablations using `MSResCNN-MLP` suggest that cardiovascular signals (`BVP`, `HR`, and `IBI`) contributed the most useful predictive information among the available wearable signal groups.


## Why This Project Matters

Sleep staging from wearable physiological signals is a challenging biomedical time-series problem because labels are imbalanced, signals are noisy, and neighboring epochs from the same participant are highly correlated. This project demonstrates practical data science and machine learning engineering skills on that realistic setting:

- wearable physiological signal processing at 64 Hz
- participant-level splitting to control leakage
- class-imbalance-aware evaluation with macro F1 and per-class metrics
- reproducible feature extraction and train-only preprocessing
- comparison of interpretable baselines and neural sequence models
- PyTorch dataset utilities for raw epochs, context windows, and sequences
- validation error analysis, signal-family ablation, label-mapping sensitivity, and transition-regularization experiments

## Dataset and Prediction Task

The DREAMT dataset was collected from 100 participants at the Duke University Health System Sleep Disorder Lab. The cohort includes participants with clinically relevant sleep-related conditions, including sleep apnea (n=56), obstructive sleep apnea (n=33), restless sleep or restless leg syndrome (n=23), difficulty breathing or gasping during sleep (n=22), excessive daytime sleepiness (n=34), and snoring (n=40). These disorder-enriched data make wearable sleep staging more challenging, but also more relevant: populations with disrupted sleep are among those most likely to benefit from accurate sleep-stage estimation outside the sleep lab.

Wearable physiological signals obtained from Empatica E4 devices include:

- `BVP` 
- `ACC_X`, `ACC_Y`, `ACC_Z`
- `TEMP`
- `EDA`
- `HR`
- `IBI`

Raw DREAMT files are not committed to GitHub and should remain local in accordance with dataset access terms and privacy requirements. Local real-data files are expected under `data/raw/`.

For reviewers without DREAMT access, the repository includes a small committed synthetic sample under `data/synthetic/`. These files are not real participant records. They use the same CSV schema expected by the project loaders and contain four synthetic participants with 31 valid 30-second epochs each. The sample is intended for pipeline smoke tests, not model-performance claims.

The prediction target is a three-class sleep-stage label:

- `Wake`
- `Non-REM`
- `REM`

Polysomnograph (PSG) labels `N1`, `N2`, and `N3` are mapped to `Non-REM`. DREAMT `data_64Hz` files include `P`, a preparation-stage label before PSG recording starts. The primary workflow excludes `P`; treating `P` as `Wake`, as in `data_100Hz` files, is implemented only as a documented sensitivity analysis.

## Evaluation Design

The project uses a fixed participant-level split: participants, not individual epochs, are assigned to train, validation, or test sets. This matters because nearby sleep epochs from the same participant are temporally correlated; splitting epochs independently would overstate generalization.

Evaluation principles:

- preprocessing and normalization statistics are fit on training participants only
- model selection, ablation studies, and error analyses use validation data only
- final held-out test evaluation was reserved until model selection was frozen
- macro F1 is the primary metric because the target classes are imbalanced

Supporting metrics include per-class precision/recall/F1, balanced accuracy, confusion matrices, participant-level macro F1, total sleep time error, and REM duration error.

## Modeling Approach

The modeling workflow was designed to progress from simple, interpretable baselines to neural architectures that incorporate raw signal morphology, engineered summary features, and longer temporal context.

**Traditional baselines**

- majority-class classifier as a sanity-check baseline
- elastic-net multinomial logistic regression on engineered epoch-level signal summaries
- XGBoost on engineered epoch-level signal summaries

**Raw-signal and sequence models**

- single-epoch 1D CNN for classifying individual 30-second raw signal windows
- temporal-context CNN using neighboring epochs around the target epoch
- CNN-GRU sequence models for explicit recurrent temporal modeling

**Fusion and temporal-context models**
- `MSResCNN-MLP`, combining a multiscale residual CNN raw-signal branch with an MLP branch for engineered features
- `MSResCNN-MLP-TCN`, using a frozen `MSResCNN-MLP` epoch encoder followed by a temporal convolutional sequence head
- transition-regularized `MSResCNN-MLP-TCN`, adding a biologically informed penalty for physiologically rare sleep-stage transitions

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
  results/              # curated summary metrics and figures; large artifacts ignored
```

The repository separates exploratory notebooks from reusable source code. Raw DREAMT data, processed local artifacts, trained checkpoints, per-epoch predictions, and large generated outputs are not committed. Curated summary CSVs and selected figures are committed under `results/` to document the main findings reported in this README; see `results/MANIFEST.md` for the artifact list and model-name crosswalk.

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

Wearable-only sleep staging remains challenging because wearable signals are indirect proxies for PSG-derived sleep stage. Unlike electroencephalography (EEG), electrooculography (EOG), and electromyography (EMG) signals used in standard sleep scoring, wearable physiological signals reflect downstream autonomic and movement patterns rather than sleep stage directly. Minority classes such as `REM` are especially difficult because they are less frequent and may be harder to distinguish from other stages using wearable signals alone.

Current limitations include:

- weaker performance on minority classes, especially `REM`
- errors near sleep-stage transition boundaries, where adjacent epochs may be physiologically ambiguous
- substantial variation in performance across participants, suggesting sensitivity to participant-specific signal patterns
- limited dataset size relative to the complexity of deep learning models, increasing the risk of overfitting to participant-level noise

Future work includes:

- identifying additional wearable sleep datasets that could be harmonized with DREAMT, especially datasets with overlapping cardiovascular signals such as `BVP`, `HR`, or `IBI`
- developing richer engineered features for traditional ML baselines, since stronger feature engineering may offer competitive performance at lower computational cost
- testing oversampling to try to improve `REM` and `Wake` classification
- further analyzing transition-boundary errors and participant-level failure modes to guide model and preprocessing improvements
