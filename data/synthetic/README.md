# Synthetic DREAMT-Compatible Sample

This folder contains four fully synthetic participant CSVs for boot-camp review
and lightweight smoke testing:

- `S901_whole_df.csv`
- `S902_whole_df.csv`
- `S903_whole_df.csv`
- `S904_whole_df.csv`

The files are not real DREAMT participant records. They use the same column
schema expected by the project's 64 Hz DREAMT loaders and contain 31 valid
30-second epochs per synthetic participant.

To run the project code against these files, copy them into local ignored raw
data storage:

```bash
mkdir -p data/raw data/interim
cp data/synthetic/S*_whole_df.csv data/raw/
```

Then create the matching local split assignment:

```bash
cat > data/interim/split_assignments.csv <<'EOF'
participant_id,split
S901,train
S902,train
S903,validation
S904,test
EOF
```

The synthetic sample is intended to verify that preprocessing, feature
construction, and dataset-loading code paths run without DREAMT access. It is
not intended for meaningful model performance claims.
