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
2. Use notebooks in order for dataset overview, training-set EDA, baselines, deep learning experiments, and error analysis.
3. Move reusable logic from notebooks into `src/`.
4. Save generated metrics and figures under `results/`.

As implementation develops, this section will include concrete commands for preprocessing, training, evaluation, and report generation.

## Results

Results are not available yet. Placeholder files are included under `results/` to show the intended outputs:

- `results/metrics.csv`
- `results/ablation_results.csv`
- `results/confusion_matrix.png`
- `results/figures/`

## Limitations

This scaffold does not yet include data, trained models, preprocessing logic, or experimental results. Future work must carefully address participant-level splitting, class imbalance, missing data, wearable signal quality, and privacy requirements.
