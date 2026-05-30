# Data Directory

Raw DREAMT data should not be committed to GitHub. Keep real dataset files local and ensure that any use of the data follows the dataset access terms and privacy requirements.

Expected local layout:

```text
data/
├── README.md
├── raw/          # local only; real DREAMT files go here
├── interim/      # local only; intermediate files
├── processed/    # local only; processed modeling files
└── synthetic/    # small synthetic dataset for illustrative purposes
```

Use `data/raw/` for local copies of real DREAMT files. Use `data/interim/` for temporary intermediate files and `data/processed/` for processed modeling files. These directories are ignored by Git so that real data and derived sensitive artifacts remain local.

A small synthetic dataset will be added later for illustrative purposes.
