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

## Methods

Planned methods include:

- Dataset inspection and integrity checks
- Training-set-only exploratory data analysis
- Signal preprocessing and label mapping
- Traditional machine learning baselines using engineered features
- PyTorch deep learning models, likely starting with a 1D CNN
- Possible sequence models such as CNN-GRU architectures
- Evaluation using accuracy, balanced accuracy, macro F1, class-specific F1 scores, and confusion matrices

These methods are placeholders for the planned workflow. The final implementation will be filled in iteratively as the project matures.

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
4. Move reusable logic from notebooks into `src/`.
5. Save generated metrics and figures under `results/`.

The dataset overview notebook writes these local intermediate summaries when raw
participant files are available:

- `data/interim/participant_summary.csv`
- `data/interim/label_mapping_summary.csv`
- `data/interim/label_mapping_summary_p_as_wake.csv`
- `data/interim/split_assignments.csv`

As implementation develops, this section will include concrete commands for preprocessing, training, evaluation, and report generation.

## Results

Results are not available yet. Placeholder files are included under `results/` to show the intended outputs:

- `results/metrics.csv`
- `results/ablation_results.csv`
- `results/confusion_matrix.png`
- `results/figures/`

## Limitations

This scaffold does not include raw data, trained models, or experimental results. Future work must carefully address participant-level splitting, class imbalance, missing data, wearable signal quality, and privacy requirements.
