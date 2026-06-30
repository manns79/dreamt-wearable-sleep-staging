"""Data loading and dataset inventory utilities for DREAMT sleep staging.

This module intentionally keeps Stage 1 work limited to raw file inventory and
basic dataset integrity summaries. It does not create predictive features,
preprocess signals, fit scalers, train models, or make train/validation/test
splits.
"""

from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Participant-level splitting should be used to reduce leakage risk. Epochs from
# the same participant should not be split across training, validation, and test
# sets when estimating generalization performance.
DEFAULT_RAW_DATA_DIR = Path("data/raw")
DEFAULT_INTERIM_DATA_DIR = Path("data/interim")
DEFAULT_PARTICIPANT_PATTERN = "S*_whole_df.csv"
DEFAULT_PARTICIPANT_SUMMARY_PATH = DEFAULT_INTERIM_DATA_DIR / "participant_summary.csv"
DEFAULT_SPLIT_ASSIGNMENTS_PATH = DEFAULT_INTERIM_DATA_DIR / "split_assignments.csv"
DEFAULT_EPOCH_INDEX_PATH = DEFAULT_INTERIM_DATA_DIR / "epoch_index.csv"
DEFAULT_PROCESSED_DATA_DIR = Path("data/processed")
DEFAULT_PREPROCESSING_METADATA_PATH = (
    DEFAULT_PROCESSED_DATA_DIR / "preprocessing_metadata.json"
)
DEFAULT_TRAIN_FEATURES_PATH = DEFAULT_PROCESSED_DATA_DIR / "features_train.csv"
DEFAULT_VALIDATION_FEATURES_PATH = DEFAULT_PROCESSED_DATA_DIR / "features_val.csv"
DEFAULT_TEST_FEATURES_PATH = DEFAULT_PROCESSED_DATA_DIR / "features_test.csv"
DEFAULT_FEATURE_PREPROCESSING_METADATA_PATH = (
    DEFAULT_PROCESSED_DATA_DIR / "feature_preprocessing_metadata.json"
)
DEFAULT_STAGE15_EMBEDDING_DIR = DEFAULT_PROCESSED_DATA_DIR / "stage15_embeddings"
STAGE15_EMBEDDING_MANIFEST = "manifest.json"
DEFAULT_PARTICIPANT_ARRAY_CACHE_DIR = (
    DEFAULT_PROCESSED_DATA_DIR / "deep" / "participants"
)
PARTICIPANT_ARRAY_CACHE_MANIFEST = "manifest.json"
DEFAULT_INVENTORY_CHUNKSIZE = 500_000
DEFAULT_EPOCH_INDEX_CHUNKSIZE = 500_000
DEFAULT_MAX_CACHED_PARTICIPANTS = 4
DEFAULT_DATASET_DTYPE = np.float32
DEFAULT_FEATURE_PREPROCESSING_CHUNKSIZE = 10_000
FEATURE_ID_COLUMNS = ["participant_id", "epoch_id", "split", "label"]

TIME_COLUMN = "TIMESTAMP"
LABEL_COLUMN = "Sleep_Stage"
EXPECTED_SIGNAL_COLUMNS = [
    "BVP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "TEMP",
    "EDA",
    "HR",
    "IBI",
]
EVENT_ANNOTATION_COLUMNS = [
    "Obstructive_Apnea",
    "Central_Apnea",
    "Hypopnea",
    "Multiple_Events",
]
EXPECTED_DREAMT_COLUMNS = [
    TIME_COLUMN,
    *EXPECTED_SIGNAL_COLUMNS,
    LABEL_COLUMN,
    *EVENT_ANNOTATION_COLUMNS,
]
LABEL_FALLBACK_TERMS = ("stage", "label", "sleep", "annotation", "psg")
PARTICIPANT_SUMMARY_COLUMNS = [
    "participant_id",
    "file_path",
    "n_rows",
    "n_columns",
    "available_columns",
    "has_expected_schema",
    "has_all_expected_columns",
    "missing_expected_columns",
    "extra_columns",
    "missing_expected_signal_columns",
    "recording_duration_seconds",
    "label_column",
    "label_column_candidates",
    "unique_label_values",
    "label_counts",
    "missing_value_counts_by_signal",
    "missing_value_percentages_by_signal",
    "signal_summary_stats",
    "event_annotation_value_counts",
    "event_annotation_unique_values",
    "error",
    *[f"has_{column}" for column in EXPECTED_SIGNAL_COLUMNS],
    *[f"has_{column}" for column in EVENT_ANNOTATION_COLUMNS],
]
SPLIT_ASSIGNMENT_COLUMNS = ["participant_id", "split"]
SPLIT_LABEL_ORDER = ["train", "validation", "test"]
TARGET_LABELS = ["Wake", "Non-REM", "REM"]
LABEL_TO_ID = {label: index for index, label in enumerate(TARGET_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
TARGET_LABEL_COLUMN_MAP = {
    "Wake": "Wake",
    "Non-REM": "Non_REM",
    "Non_REM": "Non_REM",
    "REM": "REM",
}

EPOCH_INDEX_COLUMNS = [
    "participant_id",
    "split",
    "epoch_id",
    "start_row",
    "end_row",
    "n_rows",
    "expected_n_rows",
    "start_time",
    "end_time",
    "epoch_start_offset_rows",
    "raw_label",
    "mapped_label",
    "is_valid_label",
    "is_valid_epoch",
    "exclusion_reason",
    *[f"missingness_{column}" for column in EXPECTED_SIGNAL_COLUMNS],
]


def _json_dumps(value: object) -> str:
    """Serialize nested summary values consistently for CSV output."""
    return json.dumps(value, sort_keys=True, default=str)


def list_participant_csvs(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    pattern: str = DEFAULT_PARTICIPANT_PATTERN,
) -> list[Path]:
    """Return sorted DREAMT participant CSV paths from a raw data directory.

    Parameters
    ----------
    raw_data_dir:
        Directory containing local DREAMT participant CSV files.
    pattern:
        Glob pattern used to identify participant files. Defaults to the known
        DREAMT-style file names, such as ``S002_whole_df.csv``.

    Raises
    ------
    FileNotFoundError
        If ``raw_data_dir`` does not exist.
    NotADirectoryError
        If ``raw_data_dir`` exists but is not a directory.
    """
    raw_path = Path(raw_data_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")
    if not raw_path.is_dir():
        raise NotADirectoryError(f"Raw data path is not a directory: {raw_path}")

    return sorted(path for path in raw_path.glob(pattern) if path.is_file())


def extract_participant_id(file_path: str | Path) -> str:
    """Extract a DREAMT participant ID from a file path or name.

    Examples
    --------
    ``S002_whole_df.csv`` -> ``S002``

    Raises
    ------
    ValueError
        If a participant ID cannot be parsed.
    """
    file_name = Path(file_path).name
    match = re.search(r"(?i)\b(S\d+)(?=_whole_df\.csv$|[_\-.]|$)", file_name)
    if match is None:
        raise ValueError(f"Could not parse DREAMT participant ID from: {file_name}")
    return match.group(1).upper()


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    frame_name: str,
) -> None:
    """Raise a clear error when a DataFrame is missing required columns."""
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required column(s): {missing_columns}"
        )


def _clean_participant_ids(participant_ids: Iterable[object]) -> list[str]:
    """Normalize participant IDs to stripped strings and validate uniqueness."""
    cleaned_ids: list[str] = []
    for participant_id in participant_ids:
        if pd.isna(participant_id):
            raise ValueError("Participant IDs must not contain missing values.")
        participant_text = str(participant_id).strip()
        if not participant_text:
            raise ValueError("Participant IDs must not contain empty strings.")
        cleaned_ids.append(participant_text)

    if not cleaned_ids:
        raise ValueError("At least one participant ID is required.")

    duplicate_ids = sorted(
        {
            participant_id
            for participant_id in cleaned_ids
            if cleaned_ids.count(participant_id) > 1
        }
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate participant ID(s) found: {duplicate_ids}")

    return cleaned_ids


def create_participant_split(
    participant_ids: Iterable[object],
    train_size: int = 70,
    val_size: int = 15,
    test_size: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a reproducible participant-level train/validation/test split.

    Participants are sorted before seeded shuffling so the same IDs and random
    seed produce the same split regardless of input iteration order. The
    returned DataFrame has exactly ``participant_id`` and ``split`` columns.
    """
    split_sizes = {
        "train": train_size,
        "validation": val_size,
        "test": test_size,
    }
    invalid_sizes = {
        split: size
        for split, size in split_sizes.items()
        if not isinstance(size, int) or size < 0
    }
    if invalid_sizes:
        raise ValueError(
            "Split sizes must be non-negative integers. "
            f"Invalid values: {invalid_sizes}"
        )

    cleaned_ids = _clean_participant_ids(participant_ids)
    requested_total = train_size + val_size + test_size
    if requested_total != len(cleaned_ids):
        raise ValueError(
            "Requested split sizes must match the number of participants: "
            f"train_size + val_size + test_size = {requested_total}, "
            f"but received {len(cleaned_ids)} participant IDs."
        )

    shuffled_ids = np.array(sorted(cleaned_ids), dtype=object)
    rng = np.random.default_rng(random_state)
    rng.shuffle(shuffled_ids)

    split_values = (
        ["train"] * train_size
        + ["validation"] * val_size
        + ["test"] * test_size
    )
    split_df = pd.DataFrame(
        {
            "participant_id": shuffled_ids.tolist(),
            "split": split_values,
        },
        columns=SPLIT_ASSIGNMENT_COLUMNS,
    )
    check_no_participant_overlap(split_df)
    return split_df


def check_no_participant_overlap(split_df: pd.DataFrame) -> bool:
    """Confirm each participant appears in exactly one split assignment.

    Raises
    ------
    ValueError
        If required columns are missing, participant IDs are missing or empty,
        split labels are unexpected, or any participant has duplicate or
        conflicting assignments.
    """
    _validate_required_columns(split_df, SPLIT_ASSIGNMENT_COLUMNS, "split_df")

    cleaned_df = split_df[SPLIT_ASSIGNMENT_COLUMNS].copy()
    cleaned_df["participant_id"] = cleaned_df["participant_id"].map(
        lambda value: None if pd.isna(value) else str(value).strip()
    )
    cleaned_df["split"] = cleaned_df["split"].map(
        lambda value: None if pd.isna(value) else str(value).strip()
    )

    if cleaned_df["participant_id"].isna().any() or (
        cleaned_df["participant_id"] == ""
    ).any():
        raise ValueError("Split assignments contain missing participant IDs.")
    if cleaned_df["split"].isna().any() or (cleaned_df["split"] == "").any():
        raise ValueError("Split assignments contain missing split labels.")

    unexpected_splits = sorted(set(cleaned_df["split"]) - set(SPLIT_LABEL_ORDER))
    if unexpected_splits:
        raise ValueError(
            "Split assignments contain unexpected split label(s): "
            f"{unexpected_splits}. Expected labels are {SPLIT_LABEL_ORDER}."
        )

    duplicate_rows = cleaned_df[
        cleaned_df.duplicated(subset=["participant_id"], keep=False)
    ]
    if not duplicate_rows.empty:
        conflicting_ids = sorted(
            participant_id
            for participant_id, group in duplicate_rows.groupby("participant_id")
            if group["split"].nunique() > 1
        )
        if conflicting_ids:
            raise ValueError(
                "Participant(s) assigned to multiple splits: "
                f"{conflicting_ids}"
            )

        duplicate_ids = sorted(duplicate_rows["participant_id"].unique())
        raise ValueError(
            "Duplicate split assignment(s) found for participant(s): "
            f"{duplicate_ids}"
        )

    return True


def save_split_assignments(split_df: pd.DataFrame, path: str | Path) -> None:
    """Save participant split assignments to CSV, creating parents as needed."""
    _validate_required_columns(split_df, SPLIT_ASSIGNMENT_COLUMNS, "split_df")
    output_df = split_df[SPLIT_ASSIGNMENT_COLUMNS].copy()
    check_no_participant_overlap(output_df)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)


def load_split_assignments(path: str | Path) -> pd.DataFrame:
    """Load saved participant split assignments and validate their schema."""
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split assignment CSV does not exist: {split_path}")

    split_df = pd.read_csv(split_path, dtype={"participant_id": str, "split": str})
    _validate_required_columns(split_df, SPLIT_ASSIGNMENT_COLUMNS, "split_df")
    split_df = split_df[SPLIT_ASSIGNMENT_COLUMNS].copy()
    check_no_participant_overlap(split_df)
    return split_df


def load_participant_csv(
    file_path: str | Path,
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load a participant CSV without preprocessing.

    Raises a clear error for missing files or unreadable CSVs.
    """
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Participant CSV does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Participant CSV path is not a file: {csv_path}")

    read_kwargs = {}
    if usecols is not None:
        requested_columns = set(usecols)
        read_kwargs["usecols"] = lambda column: column in requested_columns

    try:
        return pd.read_csv(csv_path, **read_kwargs)
    except Exception as exc:
        raise ValueError(f"Could not read participant CSV {csv_path}: {exc}") from exc


def _participant_file_lookup(raw_dir: str | Path) -> dict[str, Path]:
    """Map participant IDs to raw CSV paths using the shared inventory rules."""
    return {
        extract_participant_id(path): path for path in list_participant_csvs(raw_dir)
    }


def _load_valid_epoch_index(
    epoch_index: str | Path | pd.DataFrame = DEFAULT_EPOCH_INDEX_PATH,
) -> pd.DataFrame:
    """Load valid three-class epochs from a path or DataFrame."""
    if isinstance(epoch_index, pd.DataFrame):
        epoch_df = epoch_index.copy()
    else:
        epoch_path = Path(epoch_index)
        if not epoch_path.exists():
            raise FileNotFoundError(f"Epoch index CSV does not exist: {epoch_path}")
        epoch_df = pd.read_csv(epoch_path, dtype={"participant_id": str})

    required_columns = {
        "participant_id",
        "epoch_id",
        "start_row",
        "end_row",
        "split",
        "mapped_label",
        "is_valid_epoch",
    }
    missing_columns = sorted(required_columns - set(epoch_df.columns))
    if missing_columns:
        raise ValueError(f"epoch_index is missing column(s): {missing_columns}")

    valid_epochs = epoch_df[epoch_df["is_valid_epoch"].astype(bool)].copy()
    valid_epochs = valid_epochs[valid_epochs["mapped_label"].isin(TARGET_LABELS)]
    valid_epochs["participant_id"] = (
        valid_epochs["participant_id"].astype(str).str.strip()
    )
    valid_epochs["epoch_id"] = pd.to_numeric(
        valid_epochs["epoch_id"], errors="raise"
    ).astype(int)
    valid_epochs["start_row"] = pd.to_numeric(
        valid_epochs["start_row"], errors="raise"
    ).astype(int)
    valid_epochs["end_row"] = pd.to_numeric(
        valid_epochs["end_row"], errors="raise"
    ).astype(int)
    return valid_epochs.reset_index(drop=True)


def _filter_epoch_index_for_split(
    epoch_index: pd.DataFrame,
    split: str | None = None,
    participant_ids: Iterable[str] | None = None,
    max_participants: int | None = None,
) -> pd.DataFrame:
    """Filter epochs by split and optional participant debug subset."""
    output = epoch_index.copy()
    if split is not None:
        output = output[output["split"] == split].copy()

    if participant_ids is not None:
        participant_set = {
            str(participant_id).strip() for participant_id in participant_ids
        }
        output = output[output["participant_id"].isin(participant_set)].copy()

    if max_participants is not None:
        if max_participants <= 0:
            raise ValueError("max_participants must be positive when provided.")
        selected_ids = sorted(output["participant_id"].unique())[:max_participants]
        output = output[output["participant_id"].isin(selected_ids)].copy()

    return output.sort_values(["participant_id", "epoch_id"]).reset_index(drop=True)


def check_epoch_split_leakage(epoch_index: pd.DataFrame) -> bool:
    """Confirm that no participant appears in more than one split."""
    required_columns = {"participant_id", "split"}
    missing_columns = sorted(required_columns - set(epoch_index.columns))
    if missing_columns:
        raise ValueError(f"epoch_index is missing column(s): {missing_columns}")

    participant_splits = (
        epoch_index[["participant_id", "split"]]
        .dropna()
        .assign(
            participant_id=lambda df: df["participant_id"].astype(str).str.strip(),
            split=lambda df: df["split"].astype(str).str.strip(),
        )
        .drop_duplicates()
    )
    split_counts = participant_splits.groupby("participant_id")["split"].nunique()
    overlapping_ids = sorted(split_counts[split_counts > 1].index.tolist())
    if overlapping_ids:
        raise ValueError(f"Participant(s) appear in multiple splits: {overlapping_ids}")
    return True


def _raw_columns_for_channels(channels: Iterable[str]) -> list[str]:
    """Return raw CSV columns needed to build requested model channels."""
    required: list[str] = []
    for channel in channels:
        if channel == "ACC_MAG":
            for axis in ["ACC_X", "ACC_Y", "ACC_Z"]:
                if axis not in required:
                    required.append(axis)
        elif channel not in required:
            required.append(channel)
    return required


def _add_derived_channels(df: pd.DataFrame, channels: Iterable[str]) -> pd.DataFrame:
    """Add simple derived signal channels needed by deep learning datasets."""
    output = df.copy()
    if "ACC_MAG" in channels and "ACC_MAG" not in output.columns:
        missing_axes = [
            column for column in ["ACC_X", "ACC_Y", "ACC_Z"] if column not in output
        ]
        if missing_axes:
            raise ValueError(
                "Cannot compute ACC_MAG; missing raw column(s): "
                f"{missing_axes}"
            )
        axes = output[["ACC_X", "ACC_Y", "ACC_Z"]].apply(pd.to_numeric, errors="coerce")
        output["ACC_MAG"] = np.sqrt((axes**2).sum(axis=1, min_count=3))
    return output


def _participant_array_manifest_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / PARTICIPANT_ARRAY_CACHE_MANIFEST


def _participant_array_file_name(participant_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(participant_id).strip())
    if not safe_id:
        raise ValueError("Participant ID must not be empty.")
    return f"{safe_id}.npy"


def _load_participant_array_manifest(cache_dir: str | Path) -> dict[str, Any]:
    manifest_path = _participant_array_manifest_path(cache_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Participant array cache manifest does not exist: "
            f"{manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid participant array cache manifest: {manifest_path}")
    return manifest


def _participant_array_manifest_matches_channels(
    manifest: Mapping[str, object],
    channels: Iterable[str],
) -> bool:
    return list(manifest.get("channels", [])) == list(channels)


def build_participant_array_cache(
    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    output_dir: str | Path = DEFAULT_PARTICIPANT_ARRAY_CACHE_DIR,
    channels: Iterable[str] = EXPECTED_SIGNAL_COLUMNS,
    participant_ids: Iterable[str] | None = None,
    pattern: str = DEFAULT_PARTICIPANT_PATTERN,
    dtype: Any = DEFAULT_DATASET_DTYPE,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build per-participant NumPy arrays for faster deep-learning slicing.

    Each participant is stored as one ``.npy`` array shaped
    ``(rows, channels)``. Arrays are intentionally not compressed so they can be
    loaded quickly and memory-mapped during training.
    """

    channels = list(channels)
    array_dtype = np.dtype(dtype)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = _participant_array_manifest_path(output_path)

    if manifest_path.exists() and not overwrite:
        existing_manifest = _load_participant_array_manifest(output_path)
        if not _participant_array_manifest_matches_channels(
            existing_manifest,
            channels,
        ):
            raise ValueError(
                "Existing participant array cache uses different channels. "
                "Use overwrite=True or a different output_dir."
            )

    file_lookup = {
        extract_participant_id(path): path
        for path in list_participant_csvs(raw_dir, pattern=pattern)
    }
    if participant_ids is None:
        selected_ids = sorted(file_lookup)
    else:
        selected_ids = sorted(
            str(participant_id).strip() for participant_id in participant_ids
        )

    missing_ids = sorted(set(selected_ids) - set(file_lookup))
    if missing_ids:
        raise FileNotFoundError(
            "Raw CSV file(s) were not found for participant(s): "
            f"{missing_ids}"
        )

    raw_columns = _raw_columns_for_channels(channels)
    participants: dict[str, dict[str, object]] = {}
    for participant_id in selected_ids:
        file_name = _participant_array_file_name(participant_id)
        array_path = output_path / file_name

        if array_path.exists() and not overwrite:
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 2 or array.shape[1] != len(channels):
                raise ValueError(
                    "Existing participant array has unexpected shape: "
                    f"{array_path}"
                )
            n_rows = int(array.shape[0])
        else:
            raw_df = load_participant_csv(
                file_lookup[participant_id],
                usecols=raw_columns,
            )
            signal_df = _add_derived_channels(raw_df, channels)
            columns = [
                pd.to_numeric(signal_df[channel], errors="coerce").to_numpy(
                    dtype=array_dtype,
                    copy=True,
                )
                for channel in channels
            ]
            array = np.column_stack(columns).astype(array_dtype, copy=False)
            np.save(array_path, array, allow_pickle=False)
            n_rows = int(array.shape[0])

        participants[participant_id] = {
            "file": file_name,
            "n_rows": n_rows,
        }

    manifest: dict[str, object] = {
        "version": 1,
        "format": "npy_rows_by_channels",
        "channels": channels,
        "dtype": array_dtype.name,
        "participants": participants,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    return manifest


class _ParticipantSignalCache:
    """Lazy participant-level raw signal cache for epoch/window slicing."""

    def __init__(
        self,
        raw_dir: str | Path,
        channels: Iterable[str],
        max_cached_participants: int | None = DEFAULT_MAX_CACHED_PARTICIPANTS,
    ):
        if max_cached_participants is not None and max_cached_participants <= 0:
            raise ValueError(
                "max_cached_participants must be positive or None for no limit."
            )
        self.raw_dir = Path(raw_dir)
        self.channels = list(channels)
        self.max_cached_participants = max_cached_participants
        self.raw_columns = _raw_columns_for_channels(self.channels)
        self.file_lookup = _participant_file_lookup(self.raw_dir)
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.load_count = 0

    def require_participants(self, participant_ids: Iterable[str]) -> None:
        missing_ids = sorted(set(participant_ids) - set(self.file_lookup))
        if missing_ids:
            raise FileNotFoundError(
                "Raw CSV file(s) were not found for participant(s): "
                f"{missing_ids}"
            )

    def get(self, participant_id: str) -> pd.DataFrame:
        if participant_id in self._cache:
            self._cache.move_to_end(participant_id)
        else:
            self.load_count += 1
            raw_df = load_participant_csv(
                self.file_lookup[participant_id],
                usecols=self.raw_columns,
            )
            raw_df = _add_derived_channels(raw_df, self.channels)
            self._cache[participant_id] = raw_df[self.channels].copy()
            if (
                self.max_cached_participants is not None
                and len(self._cache) > self.max_cached_participants
            ):
                self._cache.popitem(last=False)
        return self._cache[participant_id]


class _ParticipantArrayCache:
    """Lazy participant-level NumPy array cache for epoch/window slicing."""

    def __init__(
        self,
        cache_dir: str | Path,
        channels: Iterable[str],
        max_cached_participants: int | None = DEFAULT_MAX_CACHED_PARTICIPANTS,
    ):
        if max_cached_participants is not None and max_cached_participants <= 0:
            raise ValueError(
                "max_cached_participants must be positive or None for no limit."
            )
        self.cache_dir = Path(cache_dir)
        self.channels = list(channels)
        self.max_cached_participants = max_cached_participants
        self.manifest = _load_participant_array_manifest(self.cache_dir)
        if not _participant_array_manifest_matches_channels(
            self.manifest,
            self.channels,
        ):
            raise ValueError(
                "Participant array cache channels do not match requested channels."
            )
        participants = self.manifest.get("participants")
        if not isinstance(participants, Mapping):
            raise ValueError(
                "Participant array cache manifest is missing participants."
            )
        self.participants = {
            str(participant_id): dict(info)
            for participant_id, info in participants.items()
        }
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.load_count = 0

    def require_participants(self, participant_ids: Iterable[str]) -> None:
        missing_ids = sorted(set(participant_ids) - set(self.participants))
        if missing_ids:
            raise FileNotFoundError(
                "Participant array cache file(s) were not found for participant(s): "
                f"{missing_ids}"
            )

    def get(self, participant_id: str) -> np.ndarray:
        if participant_id in self._cache:
            self._cache.move_to_end(participant_id)
        else:
            participant_info = self.participants[participant_id]
            array_path = self.cache_dir / str(participant_info["file"])
            if not array_path.exists():
                raise FileNotFoundError(
                    f"Participant array cache file does not exist: {array_path}"
                )
            self.load_count += 1
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 2 or array.shape[1] != len(self.channels):
                raise ValueError(
                    "Participant array cache file has unexpected shape: "
                    f"{array_path}"
                )
            self._cache[participant_id] = array
            if (
                self.max_cached_participants is not None
                and len(self._cache) > self.max_cached_participants
            ):
                self._cache.popitem(last=False)
        return self._cache[participant_id]


def _epoch_row_to_array(
    epoch_row: pd.Series,
    signal_cache: _ParticipantSignalCache | _ParticipantArrayCache,
    channels: list[str],
) -> np.ndarray:
    participant_id = str(epoch_row["participant_id"])
    start_row = int(epoch_row["start_row"])
    end_row = int(epoch_row["end_row"])
    raw_data = signal_cache.get(participant_id)
    if isinstance(raw_data, pd.DataFrame):
        epoch_df = raw_data.iloc[start_row:end_row]
        if epoch_df.empty:
            raise ValueError(
                f"Epoch slice is empty for participant {participant_id}, "
                f"epoch {epoch_row['epoch_id']}."
            )
        return (
            epoch_df[channels]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
            .T
        )

    epoch_array = raw_data[start_row:end_row]
    if epoch_array.shape[0] == 0:
        raise ValueError(
            f"Epoch slice is empty for participant {participant_id}, "
            f"epoch {epoch_row['epoch_id']}."
        )
    return np.asarray(epoch_array.T, dtype=np.float64)


def _ensure_writable_array(array: np.ndarray) -> np.ndarray:
    """Copy read-only NumPy views before handing them to tensor constructors."""
    if array.flags.writeable:
        return array
    return array.copy()


def _label_to_id(label: object) -> int:
    label_text = str(label)
    if label_text not in LABEL_TO_ID:
        raise ValueError(f"Unexpected target label: {label_text}")
    return LABEL_TO_ID[label_text]


def _feature_columns_from_frame(feature_frame: pd.DataFrame) -> list[str]:
    """Return engineered feature columns after validating identity fields."""

    missing_columns = sorted(set(FEATURE_ID_COLUMNS) - set(feature_frame.columns))
    if missing_columns:
        raise ValueError(
            f"Engineered feature table is missing column(s): {missing_columns}"
        )
    feature_columns = [
        column for column in feature_frame.columns if column not in FEATURE_ID_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("Engineered feature table has no feature columns.")
    return feature_columns


def _iter_feature_chunks(
    feature_source: str | Path | pd.DataFrame,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    """Yield feature-table chunks without materializing a full CSV during fitting."""

    if isinstance(feature_source, pd.DataFrame):
        yield feature_source
        return

    feature_path = Path(feature_source)
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Engineered feature CSV does not exist: {feature_path}"
        )
    yield from pd.read_csv(
        feature_path,
        dtype={"participant_id": str},
        chunksize=chunksize,
    )


def fit_engineered_feature_preprocessing(
    feature_source: str | Path | pd.DataFrame,
    *,
    expected_split: str = "train",
    chunksize: int = DEFAULT_FEATURE_PREPROCESSING_CHUNKSIZE,
    epsilon: float = 1e-8,
) -> dict[str, object]:
    """Fit train-only mean imputation and standardization for engineered features.

    CSV inputs are processed in chunks and only scalar per-feature summaries are
    retained. This avoids the memory-heavy median/materialization path that was
    unsuitable for local training.
    """

    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    feature_columns: list[str] | None = None
    accumulators: dict[str, dict[str, float | int]] = {}
    row_count = 0
    source_participants: set[str] = set()

    for chunk in _iter_feature_chunks(feature_source, chunksize=chunksize):
        current_columns = _feature_columns_from_frame(chunk)
        if feature_columns is None:
            feature_columns = current_columns
            accumulators = _new_channel_summary_accumulator(feature_columns)
        elif current_columns != feature_columns:
            raise ValueError("Engineered feature columns changed between CSV chunks.")

        split_values = set(chunk["split"].astype(str).str.strip())
        if split_values != {expected_split}:
            raise ValueError(
                "Engineered preprocessing must be fit on the expected split only: "
                f"expected {expected_split!r}, found {sorted(split_values)}."
            )

        row_count += len(chunk)
        source_participants.update(
            chunk["participant_id"].astype(str).str.strip().tolist()
        )
        for column in feature_columns:
            _update_channel_summary(accumulators[column], chunk[column])

    if not feature_columns or row_count == 0:
        raise ValueError("Cannot fit engineered preprocessing on an empty table.")

    stats = _finalize_streaming_channel_preprocessing_stats(
        accumulators,
        epsilon=epsilon,
    )
    stats.update(
        {
            "feature_columns": feature_columns,
            "fit_scope": expected_split,
            "source_participants": sorted(source_participants),
            "n_rows_fit": int(row_count),
        }
    )
    stats.pop("channels", None)
    return stats


def apply_engineered_feature_preprocessing(
    feature_frame: pd.DataFrame,
    stats: Mapping[str, object],
    *,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Apply fitted mean imputation and standardization to engineered features."""

    feature_columns = [str(column) for column in stats.get("feature_columns", [])]
    if not feature_columns:
        raise ValueError("Feature preprocessing metadata has no feature_columns.")
    if stats.get("imputation_strategy") != "mean":
        raise ValueError("Engineered feature preprocessing must use mean imputation.")

    missing_columns = sorted(set(feature_columns) - set(feature_frame.columns))
    if missing_columns:
        raise ValueError(
            f"Engineered feature table is missing fitted column(s): {missing_columns}"
        )
    unexpected_columns = [
        column
        for column in _feature_columns_from_frame(feature_frame)
        if column not in feature_columns
    ]
    if unexpected_columns:
        raise ValueError(
            "Engineered feature table has unexpected feature column(s): "
            f"{unexpected_columns}"
        )

    means = stats.get("mean")
    stds = stats.get("std")
    if not isinstance(means, Mapping) or not isinstance(stds, Mapping):
        raise ValueError("Feature preprocessing metadata must contain mean and std.")

    matrix = np.empty(
        (len(feature_frame), len(feature_columns)),
        dtype=np.dtype(dtype),
    )
    for column_index, column in enumerate(feature_columns):
        values = pd.to_numeric(feature_frame[column], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=True,
        )
        mean = float(means[column])
        std = float(stds[column])
        values[np.isnan(values)] = mean
        matrix[:, column_index] = ((values - mean) / std).astype(
            matrix.dtype,
            copy=False,
        )
    return matrix


class DreamtEpochDataset:
    """PyTorch dataset for single DREAMT sleep epochs.

    Items are returned as ``(x, y)`` where ``x`` has shape
    ``(channels, timepoints)`` and dtype ``torch.float32`` by default. Labels
    are integer class IDs using ``LABEL_TO_ID``.
    """

    def __init__(
        self,
        raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
        epoch_index: str | Path | pd.DataFrame = DEFAULT_EPOCH_INDEX_PATH,
        split: str | None = None,
        channels: Iterable[str] = EXPECTED_SIGNAL_COLUMNS,
        preprocessing_stats: dict[str, object] | None = None,
        participant_ids: Iterable[str] | None = None,
        max_participants: int | None = None,
        max_cached_participants: int | None = DEFAULT_MAX_CACHED_PARTICIPANTS,
        participant_array_cache_dir: str | Path | None = None,
        dtype: Any = DEFAULT_DATASET_DTYPE,
    ):
        self.channels = list(channels)
        self.preprocessing_stats = preprocessing_stats
        self.dtype = np.dtype(dtype)
        valid_epochs = _load_valid_epoch_index(epoch_index)
        check_epoch_split_leakage(valid_epochs)
        self.epoch_index = _filter_epoch_index_for_split(
            valid_epochs,
            split=split,
            participant_ids=participant_ids,
            max_participants=max_participants,
        )
        if participant_array_cache_dir is None:
            self.signal_cache = _ParticipantSignalCache(
                raw_dir,
                self.channels,
                max_cached_participants=max_cached_participants,
            )
        else:
            self.signal_cache = _ParticipantArrayCache(
                participant_array_cache_dir,
                self.channels,
                max_cached_participants=max_cached_participants,
            )
        self.signal_cache.require_participants(self.epoch_index["participant_id"])

    def __len__(self) -> int:
        return len(self.epoch_index)

    @property
    def participants(self) -> set[str]:
        return set(self.epoch_index["participant_id"].astype(str))

    def get_epoch_array(self, position: int) -> np.ndarray:
        row = self.epoch_index.iloc[position]
        x = _epoch_row_to_array(row, self.signal_cache, self.channels)
        if self.preprocessing_stats is not None:
            from src.preprocessing import apply_normalization

            x = apply_normalization(x, self.preprocessing_stats)
        return _ensure_writable_array(x.astype(self.dtype, copy=False))

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        import torch

        row = self.epoch_index.iloc[index]
        x = self.get_epoch_array(index)
        y = _label_to_id(row["mapped_label"])
        return (
            torch.as_tensor(x),
            torch.tensor(y, dtype=torch.long),
        )


class DreamtFeatureFusionDataset:
    """Pair one raw DREAMT epoch with its aligned engineered feature vector."""

    def __init__(
        self,
        raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
        epoch_index: str | Path | pd.DataFrame = DEFAULT_EPOCH_INDEX_PATH,
        feature_table: str | Path | pd.DataFrame = DEFAULT_TRAIN_FEATURES_PATH,
        split: str | None = None,
        channels: Iterable[str] = EXPECTED_SIGNAL_COLUMNS,
        preprocessing_stats: dict[str, object] | None = None,
        feature_preprocessing_stats: Mapping[str, object] | None = None,
        participant_ids: Iterable[str] | None = None,
        max_participants: int | None = None,
        max_cached_participants: int | None = DEFAULT_MAX_CACHED_PARTICIPANTS,
        participant_array_cache_dir: str | Path | None = None,
        dtype: Any = DEFAULT_DATASET_DTYPE,
    ):
        if feature_preprocessing_stats is None:
            raise ValueError("feature_preprocessing_stats are required.")

        self.raw_dataset = DreamtEpochDataset(
            raw_dir=raw_dir,
            epoch_index=epoch_index,
            split=split,
            channels=channels,
            preprocessing_stats=preprocessing_stats,
            participant_ids=participant_ids,
            max_participants=max_participants,
            max_cached_participants=max_cached_participants,
            participant_array_cache_dir=participant_array_cache_dir,
            dtype=dtype,
        )
        self.channels = self.raw_dataset.channels
        self.epoch_index = self.raw_dataset.epoch_index
        self.signal_cache = self.raw_dataset.signal_cache
        self.dtype = self.raw_dataset.dtype
        self.feature_preprocessing_stats = dict(feature_preprocessing_stats)

        if isinstance(feature_table, pd.DataFrame):
            feature_frame = feature_table.copy()
        else:
            feature_path = Path(feature_table)
            if not feature_path.exists():
                raise FileNotFoundError(
                    f"Engineered feature CSV does not exist: {feature_path}"
                )
            feature_header = pd.read_csv(feature_path, nrows=0)
            feature_dtypes = {
                column: np.float32
                for column in _feature_columns_from_frame(feature_header)
            }
            feature_dtypes["participant_id"] = str
            feature_frame = pd.read_csv(
                feature_path,
                dtype=feature_dtypes,
            )

        table_feature_columns = _feature_columns_from_frame(feature_frame)
        feature_frame["participant_id"] = (
            feature_frame["participant_id"].astype(str).str.strip()
        )
        feature_frame["epoch_id"] = pd.to_numeric(
            feature_frame["epoch_id"],
            errors="raise",
        ).astype(int)
        feature_frame["split"] = feature_frame["split"].astype(str).str.strip()
        feature_frame["label"] = feature_frame["label"].astype(str).str.strip()

        duplicate_mask = feature_frame.duplicated(
            ["participant_id", "epoch_id"],
            keep=False,
        )
        if duplicate_mask.any():
            duplicates = feature_frame.loc[
                duplicate_mask,
                ["participant_id", "epoch_id"],
            ].to_dict("records")
            raise ValueError(
                f"Engineered feature identities are duplicated: {duplicates}"
            )

        if split is not None:
            found_splits = set(feature_frame["split"])
            if found_splits != {split}:
                raise ValueError(
                    "Engineered feature split does not match the requested split: "
                    f"expected {split!r}, found {sorted(found_splits)}."
                )

        raw_identity = self.epoch_index[
            ["participant_id", "epoch_id", "split", "mapped_label"]
        ].copy()
        raw_identity["participant_id"] = (
            raw_identity["participant_id"].astype(str).str.strip()
        )
        raw_identity["epoch_id"] = pd.to_numeric(
            raw_identity["epoch_id"],
            errors="raise",
        ).astype(int)
        raw_identity = raw_identity.rename(
            columns={
                "split": "raw_split",
                "mapped_label": "raw_label",
            }
        )

        selected_participants = set(raw_identity["participant_id"])
        selected_features = feature_frame[
            feature_frame["participant_id"].isin(selected_participants)
        ].copy()
        aligned = raw_identity.merge(
            selected_features,
            on=["participant_id", "epoch_id"],
            how="outer",
            indicator=True,
            validate="one_to_one",
            sort=False,
        )
        missing_features = aligned.loc[
            aligned["_merge"] == "left_only",
            ["participant_id", "epoch_id"],
        ]
        extra_features = aligned.loc[
            aligned["_merge"] == "right_only",
            ["participant_id", "epoch_id"],
        ]
        if not missing_features.empty or not extra_features.empty:
            raise ValueError(
                "Raw and engineered epoch identities do not align: "
                f"missing_features={missing_features.to_dict('records')}, "
                f"extra_features={extra_features.to_dict('records')}."
            )

        aligned = aligned.loc[aligned["_merge"] == "both"].drop(columns="_merge")
        split_mismatch = aligned["raw_split"].astype(str) != aligned["split"]
        if split_mismatch.any():
            raise ValueError("Raw and engineered split values do not agree.")
        label_mismatch = aligned["raw_label"].astype(str) != aligned["label"]
        if label_mismatch.any():
            raise ValueError("Raw and engineered label values do not agree.")

        aligned_lookup = aligned.set_index(["participant_id", "epoch_id"])
        ordered_identity = pd.MultiIndex.from_frame(
            raw_identity[["participant_id", "epoch_id"]]
        )
        aligned_feature_frame = aligned_lookup.loc[ordered_identity].reset_index()
        aligned_feature_frame = aligned_feature_frame[
            ["participant_id", "epoch_id", "split", "label", *table_feature_columns]
        ]
        self.engineered_features = apply_engineered_feature_preprocessing(
            aligned_feature_frame,
            self.feature_preprocessing_stats,
            dtype=np.float32,
        )
        self.feature_columns = [
            str(column)
            for column in self.feature_preprocessing_stats["feature_columns"]
        ]

    def __len__(self) -> int:
        return len(self.raw_dataset)

    @property
    def participants(self) -> set[str]:
        return self.raw_dataset.participants

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        import torch

        raw_x, y = self.raw_dataset[index]
        engineered_x = torch.as_tensor(self.engineered_features[index])
        return raw_x, engineered_x, y


class DreamtContextDataset(DreamtEpochDataset):
    """PyTorch dataset for CNN temporal-context windows.

    Context is concatenated along the time axis. For ``context_radius=2``,
    each item has shape ``(channels, 5 * timepoints)``. Edge epochs are dropped
    and windows never cross participant boundaries.
    """

    def __init__(
        self,
        raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
        epoch_index: str | Path | pd.DataFrame = DEFAULT_EPOCH_INDEX_PATH,
        split: str | None = None,
        channels: Iterable[str] = EXPECTED_SIGNAL_COLUMNS,
        preprocessing_stats: dict[str, object] | None = None,
        context_radius: int = 2,
        participant_ids: Iterable[str] | None = None,
        max_participants: int | None = None,
        max_cached_participants: int | None = DEFAULT_MAX_CACHED_PARTICIPANTS,
        participant_array_cache_dir: str | Path | None = None,
        dtype: Any = DEFAULT_DATASET_DTYPE,
    ):
        if context_radius < 0:
            raise ValueError("context_radius must be non-negative.")
        self.context_radius = int(context_radius)
        super().__init__(
            raw_dir=raw_dir,
            epoch_index=epoch_index,
            split=split,
            channels=channels,
            preprocessing_stats=preprocessing_stats,
            participant_ids=participant_ids,
            max_participants=max_participants,
            max_cached_participants=max_cached_participants,
            participant_array_cache_dir=participant_array_cache_dir,
            dtype=dtype,
        )
        self.window_positions = self._build_window_positions()

    def _build_window_positions(self) -> list[list[int]]:
        if self.context_radius == 0:
            return [[index] for index in range(len(self.epoch_index))]

        windows: list[list[int]] = []
        for _, group in self.epoch_index.groupby("participant_id", sort=False):
            positions = list(group.index)
            epoch_ids = group["epoch_id"].to_numpy(dtype=int)
            stop = len(group) - self.context_radius
            for local_index in range(self.context_radius, stop):
                local_window = range(
                    local_index - self.context_radius,
                    local_index + self.context_radius + 1,
                )
                window_epoch_ids = epoch_ids[list(local_window)]
                expected_ids = np.arange(
                    epoch_ids[local_index] - self.context_radius,
                    epoch_ids[local_index] + self.context_radius + 1,
                )
                if np.array_equal(window_epoch_ids, expected_ids):
                    windows.append([positions[i] for i in local_window])
        return windows

    def __len__(self) -> int:
        return len(self.window_positions)

    @property
    def center_positions(self) -> list[int]:
        """Return epoch-index positions used as context-window targets."""

        return [positions[self.context_radius] for positions in self.window_positions]

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        positions = self.window_positions[index]
        arrays = [self.get_epoch_array(position) for position in positions]
        x = np.concatenate(arrays, axis=1).astype(self.dtype, copy=False)
        center_position = positions[self.context_radius]
        y = _label_to_id(self.epoch_index.iloc[center_position]["mapped_label"])
        return (
            torch.as_tensor(x),
            torch.tensor(y, dtype=torch.long),
        )


class DreamtSequenceDataset(DreamtEpochDataset):
    """PyTorch dataset for CNN-GRU epoch sequences.

    Items have shape ``(sequence_length, channels, timepoints)``. With
    ``label_mode="many_to_one"``, the target is the final epoch label by
    default. With ``label_mode="many_to_many"``, the target is a vector of
    labels for every epoch in the sequence. When ``return_sample_weights`` is
    enabled for many-to-many training, each item also returns one loss weight
    per sequence position.
    """

    def __init__(
        self,
        raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
        epoch_index: str | Path | pd.DataFrame = DEFAULT_EPOCH_INDEX_PATH,
        split: str | None = None,
        channels: Iterable[str] = EXPECTED_SIGNAL_COLUMNS,
        preprocessing_stats: dict[str, object] | None = None,
        sequence_length: int = 5,
        stride: int = 1,
        label_mode: str = "many_to_one",
        target_position: str = "last",
        return_sample_weights: bool = False,
        sample_weight_mode: str = "none",
        participant_ids: Iterable[str] | None = None,
        max_participants: int | None = None,
        max_cached_participants: int | None = DEFAULT_MAX_CACHED_PARTICIPANTS,
        participant_array_cache_dir: str | Path | None = None,
        dtype: Any = DEFAULT_DATASET_DTYPE,
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if label_mode not in {"many_to_one", "many_to_many"}:
            raise ValueError("label_mode must be 'many_to_one' or 'many_to_many'.")
        if target_position not in {"first", "center", "last"}:
            raise ValueError("target_position must be 'first', 'center', or 'last'.")
        if sample_weight_mode not in {"none", "inverse_epoch_coverage"}:
            raise ValueError(
                "sample_weight_mode must be 'none' or 'inverse_epoch_coverage'."
            )
        if return_sample_weights and label_mode != "many_to_many":
            raise ValueError(
                "return_sample_weights is only supported for many_to_many labels."
            )

        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.label_mode = label_mode
        self.target_position = target_position
        self.return_sample_weights = bool(return_sample_weights)
        self.sample_weight_mode = sample_weight_mode
        super().__init__(
            raw_dir=raw_dir,
            epoch_index=epoch_index,
            split=split,
            channels=channels,
            preprocessing_stats=preprocessing_stats,
            participant_ids=participant_ids,
            max_participants=max_participants,
            max_cached_participants=max_cached_participants,
            participant_array_cache_dir=participant_array_cache_dir,
            dtype=dtype,
        )
        self.sequence_positions = self._build_sequence_positions()
        self._sample_weights_by_position = self._build_sample_weights_by_position()

    def _build_sequence_positions(self) -> list[list[int]]:
        sequences: list[list[int]] = []
        for _, group in self.epoch_index.groupby("participant_id", sort=False):
            positions = list(group.index)
            epoch_ids = group["epoch_id"].to_numpy(dtype=int)
            for start in range(0, len(group) - self.sequence_length + 1, self.stride):
                stop = start + self.sequence_length
                window_epoch_ids = epoch_ids[start:stop]
                expected_ids = np.arange(
                    window_epoch_ids[0],
                    window_epoch_ids[0] + self.sequence_length,
                )
                if np.array_equal(window_epoch_ids, expected_ids):
                    sequences.append(positions[start:stop])
        return sequences

    def __len__(self) -> int:
        return len(self.sequence_positions)

    def _target_index(self) -> int:
        if self.target_position == "first":
            return 0
        if self.target_position == "center":
            return self.sequence_length // 2
        return self.sequence_length - 1

    def _build_sample_weights_by_position(self) -> dict[int, float]:
        if self.sample_weight_mode == "none":
            return {}

        counts: Counter[int] = Counter(
            int(position)
            for positions in self.sequence_positions
            for position in positions
        )
        return {position: 1.0 / count for position, count in counts.items()}

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        positions = self.sequence_positions[index]
        arrays = [self.get_epoch_array(position) for position in positions]
        x = np.stack(arrays, axis=0).astype(self.dtype, copy=False)
        labels = [
            _label_to_id(self.epoch_index.iloc[position]["mapped_label"])
            for position in positions
        ]
        if self.label_mode == "many_to_many":
            y: Any = torch.as_tensor(labels, dtype=torch.long)
        else:
            y = torch.tensor(labels[self._target_index()], dtype=torch.long)
        if not self.return_sample_weights:
            return torch.as_tensor(x), y

        weights = [
            self._sample_weights_by_position[int(position)]
            for position in positions
        ]
        return torch.as_tensor(x), y, torch.as_tensor(weights, dtype=torch.float32)


class DreamtEmbeddingSequenceDataset:
    """Many-to-many sequences over cached frozen epoch embeddings."""

    def __init__(
        self,
        embedding_path: str | Path,
        epoch_index_path: str | Path,
        sequence_length: int = 31,
        stride: int = 1,
        label_mode: str = "many_to_many",
        target_position: str = "center",
        return_sample_weights: bool = True,
        sample_weight_mode: str = "inverse_epoch_coverage",
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if label_mode not in {"many_to_one", "many_to_many"}:
            raise ValueError("label_mode must be 'many_to_one' or 'many_to_many'.")
        if target_position not in {"first", "center", "last"}:
            raise ValueError("target_position must be 'first', 'center', or 'last'.")
        if sample_weight_mode not in {"none", "inverse_epoch_coverage"}:
            raise ValueError(
                "sample_weight_mode must be 'none' or 'inverse_epoch_coverage'."
            )
        if return_sample_weights and label_mode != "many_to_many":
            raise ValueError(
                "return_sample_weights is only supported for many_to_many labels."
            )

        embedding_file = Path(embedding_path)
        index_file = Path(epoch_index_path)
        if not embedding_file.exists():
            raise FileNotFoundError(
                f"Embedding array does not exist: {embedding_file}"
            )
        if not index_file.exists():
            raise FileNotFoundError(
                f"Embedding epoch index does not exist: {index_file}"
            )

        self.embeddings = np.load(embedding_file, mmap_mode="r")
        if self.embeddings.ndim != 2:
            raise ValueError("Embedding array must have shape (epochs, features).")
        self.epoch_index = pd.read_csv(
            index_file,
            dtype={"participant_id": str},
        )
        required_columns = {
            "participant_id",
            "epoch_id",
            "split",
            "mapped_label",
        }
        missing_columns = sorted(required_columns - set(self.epoch_index.columns))
        if missing_columns:
            raise ValueError(
                f"Embedding epoch index is missing column(s): {missing_columns}"
            )
        if len(self.epoch_index) != len(self.embeddings):
            raise ValueError(
                "Embedding rows and epoch-index rows do not match: "
                f"{len(self.embeddings)} != {len(self.epoch_index)}."
            )
        if self.epoch_index.duplicated(["participant_id", "epoch_id"]).any():
            raise ValueError("Embedding epoch identities must be unique.")
        invalid_labels = sorted(
            set(self.epoch_index["mapped_label"]) - set(TARGET_LABELS)
        )
        if invalid_labels:
            raise ValueError(
                f"Embedding epoch index has invalid label(s): {invalid_labels}"
            )

        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.label_mode = label_mode
        self.target_position = target_position
        self.return_sample_weights = bool(return_sample_weights)
        self.sample_weight_mode = sample_weight_mode
        self.sequence_positions = self._build_sequence_positions()
        self._sample_weights_by_position = self._build_sample_weights_by_position()

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def participants(self) -> set[str]:
        return set(self.epoch_index["participant_id"].astype(str))

    def _build_sequence_positions(self) -> list[list[int]]:
        sequences: list[list[int]] = []
        for _, group in self.epoch_index.groupby("participant_id", sort=False):
            positions = group.index.to_numpy(dtype=int)
            epoch_ids = group["epoch_id"].to_numpy(dtype=int)
            for start in range(
                0,
                len(group) - self.sequence_length + 1,
                self.stride,
            ):
                stop = start + self.sequence_length
                window_epoch_ids = epoch_ids[start:stop]
                expected_ids = np.arange(
                    window_epoch_ids[0],
                    window_epoch_ids[0] + self.sequence_length,
                )
                if np.array_equal(window_epoch_ids, expected_ids):
                    sequences.append(positions[start:stop].tolist())
        return sequences

    def _target_index(self) -> int:
        if self.target_position == "first":
            return 0
        if self.target_position == "center":
            return self.sequence_length // 2
        return self.sequence_length - 1

    def _build_sample_weights_by_position(self) -> dict[int, float]:
        if self.sample_weight_mode == "none":
            return {}
        counts: Counter[int] = Counter(
            int(position)
            for positions in self.sequence_positions
            for position in positions
        )
        return {position: 1.0 / count for position, count in counts.items()}

    def __len__(self) -> int:
        return len(self.sequence_positions)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        import torch

        positions = self.sequence_positions[index]
        x = np.array(self.embeddings[positions], dtype=np.float32, copy=True)
        labels = [
            _label_to_id(self.epoch_index.iloc[position]["mapped_label"])
            for position in positions
        ]
        if self.label_mode == "many_to_many":
            y: Any = torch.as_tensor(labels, dtype=torch.long)
        else:
            y = torch.tensor(labels[self._target_index()], dtype=torch.long)
        if not self.return_sample_weights:
            return torch.as_tensor(x), y

        weights = [
            self._sample_weights_by_position[int(position)]
            for position in positions
        ]
        return (
            torch.as_tensor(x),
            y,
            torch.as_tensor(weights, dtype=torch.float32),
        )


def fit_normalization_stats(
    train_dataset_or_files: DreamtEpochDataset | Iterable[str | Path],
    channels: Iterable[str] = EXPECTED_SIGNAL_COLUMNS,
    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    epoch_index: str | Path | pd.DataFrame = DEFAULT_EPOCH_INDEX_PATH,
    split: str = "train",
    max_participants: int | None = None,
) -> dict[str, object]:
    """Fit train-only imputation and standardization metadata for CNN tensors.

    The fitting pass streams one participant at a time and keeps only scalar
    channel summaries in memory. Missing values are imputed with train-only
    channel means, which avoids materializing all samples just to compute
    medians.
    """

    channels = list(channels)

    if isinstance(train_dataset_or_files, DreamtEpochDataset):
        dataset = train_dataset_or_files
        channels = list(dataset.channels)
    else:
        participant_ids = [
            extract_participant_id(path) for path in train_dataset_or_files
        ]
        dataset = DreamtEpochDataset(
            raw_dir=raw_dir,
            epoch_index=epoch_index,
            split=split,
            channels=channels,
            participant_ids=participant_ids,
            max_participants=max_participants,
        )

    if len(dataset) == 0:
        raise ValueError("Cannot fit normalization stats on an empty training dataset.")

    stats = _fit_streaming_channel_preprocessing_stats(dataset, channels)
    stats["source_participants"] = sorted(dataset.participants)
    stats["n_epochs_fit"] = int(len(dataset))
    return stats


def _new_channel_summary_accumulator(
    channels: Iterable[str],
) -> dict[str, dict[str, float | int]]:
    """Create scalar accumulators for streaming channel statistics."""
    return {
        channel: {
            "valid_count": 0,
            "missing_count": 0,
            "sum": 0.0,
            "sum_squares": 0.0,
        }
        for channel in channels
    }


def _update_channel_summary(
    accumulator: dict[str, float | int],
    values: pd.Series,
) -> None:
    """Update one channel accumulator from a raw epoch slice."""
    numeric = pd.to_numeric(values, errors="coerce")
    missing_count = int(numeric.isna().sum())
    valid = numeric.dropna()

    accumulator["missing_count"] = int(accumulator["missing_count"]) + missing_count
    if valid.empty:
        return

    array = valid.to_numpy(dtype=np.float64, copy=False)
    accumulator["valid_count"] = int(accumulator["valid_count"]) + int(array.size)
    accumulator["sum"] = float(accumulator["sum"]) + float(array.sum())
    accumulator["sum_squares"] = float(accumulator["sum_squares"]) + float(
        np.dot(array, array)
    )


def _finalize_streaming_channel_preprocessing_stats(
    accumulators: dict[str, dict[str, float | int]],
    epsilon: float,
) -> dict[str, object]:
    """Build preprocessing metadata from streaming channel summaries."""
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    valid_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}

    for channel, accumulator in accumulators.items():
        valid_count = int(accumulator["valid_count"])
        if valid_count == 0:
            raise ValueError(
                f"Cannot fit preprocessing stats for {channel}; all values are missing."
            )

        mean = float(accumulator["sum"]) / valid_count
        second_moment = float(accumulator["sum_squares"]) / valid_count
        variance = max(0.0, second_moment - mean**2)
        std = float(np.sqrt(variance))
        if not pd.notna(std) or std < epsilon:
            std = float(epsilon)

        means[channel] = mean
        stds[channel] = std
        valid_counts[channel] = valid_count
        missing_counts[channel] = int(accumulator["missing_count"])

    return {
        "channels": list(accumulators),
        "normalization": "standardization",
        "imputation_strategy": "mean",
        "epsilon": float(epsilon),
        "mean": means,
        "std": stds,
        "valid_count": valid_counts,
        "missing_count": missing_counts,
        "fit_scope": "train",
    }


def _fit_streaming_channel_preprocessing_stats(
    dataset: DreamtEpochDataset,
    channels: list[str],
) -> dict[str, object]:
    """Fit channel stats by loading and releasing one participant at a time."""
    from src.preprocessing import DEFAULT_NORMALIZATION_EPSILON

    accumulators = _new_channel_summary_accumulator(channels)

    for participant_id, participant_epochs in dataset.epoch_index.groupby(
        "participant_id",
        sort=True,
    ):
        participant_id = str(participant_id)
        signal_data = dataset.signal_cache.get(participant_id)

        for _, epoch_row in participant_epochs.iterrows():
            start_row = int(epoch_row["start_row"])
            end_row = int(epoch_row["end_row"])
            if isinstance(signal_data, pd.DataFrame):
                epoch_df = signal_data.iloc[start_row:end_row]
                if epoch_df.empty:
                    raise ValueError(
                        f"Epoch slice is empty for participant {participant_id}, "
                        f"epoch {epoch_row['epoch_id']}."
                    )
                for channel in channels:
                    _update_channel_summary(accumulators[channel], epoch_df[channel])
            else:
                epoch_array = signal_data[start_row:end_row]
                if epoch_array.shape[0] == 0:
                    raise ValueError(
                        f"Epoch slice is empty for participant {participant_id}, "
                        f"epoch {epoch_row['epoch_id']}."
                    )
                for channel_index, channel in enumerate(channels):
                    _update_channel_summary(
                        accumulators[channel],
                        pd.Series(epoch_array[:, channel_index]),
                    )

    return _finalize_streaming_channel_preprocessing_stats(
        accumulators,
        epsilon=DEFAULT_NORMALIZATION_EPSILON,
    )


def apply_normalization(x: Any, stats: dict[str, object]) -> Any:
    """Compatibility wrapper for preprocessing.apply_normalization."""
    from src.preprocessing import apply_normalization as _apply_normalization

    return _apply_normalization(x, stats)


def save_preprocessing_metadata(
    stats: dict[str, object],
    path: str | Path = DEFAULT_PREPROCESSING_METADATA_PATH,
) -> None:
    """Compatibility wrapper for preprocessing.save_preprocessing_metadata."""
    from src.preprocessing import save_preprocessing_metadata as _save

    _save(stats, path)


def load_preprocessing_metadata(
    path: str | Path = DEFAULT_PREPROCESSING_METADATA_PATH,
) -> dict[str, object]:
    """Compatibility wrapper for preprocessing.load_preprocessing_metadata."""
    from src.preprocessing import load_preprocessing_metadata as _load

    return _load(path)


def create_dataloaders(
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Any,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Create PyTorch DataLoaders for train/validation/test datasets."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    from torch.utils.data import DataLoader

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
        ),
        "validation": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def build_epoch_index(
    raw_dir: str | Path,
    split_assignments_path: str | Path = DEFAULT_SPLIT_ASSIGNMENTS_PATH,
    output_path: str | Path = DEFAULT_EPOCH_INDEX_PATH,
    sampling_rate_hz: int = 64,
    epoch_length_seconds: int = 30,
    missingness_threshold: float = 0.20,
    pattern: str = DEFAULT_PARTICIPANT_PATTERN,
    chunksize: int | None = DEFAULT_EPOCH_INDEX_CHUNKSIZE,
    p_as_wake: bool = False,
    included_splits: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build and save a participant-level split-aware sleep epoch index.

    The raw DREAMT files are processed one participant at a time. No scaling,
    feature extraction, context-window construction, or model training occurs in
    this function. Participants present in ``raw_dir`` must have an assignment
    in ``split_assignments_path`` so downstream work cannot silently create
    epochs without train/validation/test membership. Raw participant CSVs are
    streamed in chunks and limited to timestamp, label, and expected signal
    columns. Set ``p_as_wake=True`` only for the documented sensitivity analysis
    that maps preparation-stage P labels to Wake.
    """
    from src.preprocessing import (
        apply_epoch_inclusion_rules,
        infer_epoch_start_offset_from_label_chunks,
        segment_participant_chunks_into_epochs,
    )

    if chunksize is not None and (
        not isinstance(chunksize, int) or chunksize <= 0
    ):
        raise ValueError("chunksize must be a positive integer or None.")

    split_df = load_split_assignments(split_assignments_path)
    split_lookup = dict(
        zip(
            split_df["participant_id"].astype(str).str.strip().str.upper(),
            split_df["split"].astype(str).str.strip(),
            strict=True,
        )
    )
    included_split_set = (
        {str(split).strip() for split in included_splits}
        if included_splits is not None
        else None
    )
    participant_files = list_participant_csvs(raw_dir, pattern=pattern)

    epoch_frames: list[pd.DataFrame] = []
    participants_without_split: list[str] = []

    for file_path in participant_files:
        participant_id = extract_participant_id(file_path)
        split = split_lookup.get(participant_id)
        if split is None:
            participants_without_split.append(participant_id)
            continue
        if included_split_set is not None and split not in included_split_set:
            continue

        columns = _read_csv_header(file_path)
        requested_columns = [TIME_COLUMN, LABEL_COLUMN, *EXPECTED_SIGNAL_COLUMNS]
        expected_n_rows = sampling_rate_hz * epoch_length_seconds
        label_chunks = _iter_inventory_chunks(
            file_path,
            columns=columns,
            requested_columns=[LABEL_COLUMN],
            chunksize=chunksize,
        )
        epoch_start_offset_rows = infer_epoch_start_offset_from_label_chunks(
            label_chunks,
            expected_n_rows=expected_n_rows,
        )
        chunks = _iter_inventory_chunks(
            file_path,
            columns=columns,
            requested_columns=requested_columns,
            chunksize=chunksize,
        )
        participant_epochs = segment_participant_chunks_into_epochs(
            chunks,
            participant_id=participant_id,
            sampling_rate_hz=sampling_rate_hz,
            epoch_length_seconds=epoch_length_seconds,
            signal_columns=EXPECTED_SIGNAL_COLUMNS,
            epoch_start_offset_rows=epoch_start_offset_rows,
            p_as_wake=p_as_wake,
        )
        participant_epochs.insert(1, "split", split)
        participant_epochs = apply_epoch_inclusion_rules(
            participant_epochs,
            missingness_threshold=missingness_threshold,
        )
        epoch_frames.append(participant_epochs)

    if participants_without_split:
        raise ValueError(
            "Participant CSV file(s) missing from split assignments: "
            f"{sorted(participants_without_split)}"
        )

    if epoch_frames:
        epoch_index = pd.concat(epoch_frames, ignore_index=True)
    else:
        epoch_index = pd.DataFrame(columns=EPOCH_INDEX_COLUMNS)

    missingness_columns = [
        column for column in epoch_index.columns if column.startswith("missingness_")
    ]
    ordered_columns = [
        *EPOCH_INDEX_COLUMNS,
        *[
            column
            for column in [
                "standardized_label",
                "label_issue",
                "has_timestamp_discontinuity",
            ]
            if column in epoch_index.columns
        ],
        *[
            column
            for column in missingness_columns
            if column not in EPOCH_INDEX_COLUMNS
        ],
    ]
    epoch_index = epoch_index[
        [column for column in ordered_columns if column in epoch_index.columns]
    ]

    output_csv = Path(output_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    epoch_index.to_csv(output_csv, index=False)
    print(
        f"Saved {len(epoch_index)} epoch row(s) for "
        f"{epoch_index['participant_id'].nunique() if not epoch_index.empty else 0} "
        f"participant(s) to {output_csv}."
    )

    return epoch_index


def identify_label_columns(
    columns: Iterable[str] | pd.DataFrame,
) -> str | list[str] | None:
    """Identify likely sleep-stage label columns.

    ``Sleep_Stage`` is returned when present. Otherwise, this function performs
    conservative case-insensitive matching against label-related terms. Multiple
    fallback matches are returned as a list so ambiguity remains visible.
    """
    if isinstance(columns, pd.DataFrame):
        column_names = list(columns.columns)
    else:
        column_names = list(columns)

    if LABEL_COLUMN in column_names:
        return LABEL_COLUMN

    candidates = [
        column
        for column in column_names
        if any(term in column.lower() for term in LABEL_FALLBACK_TERMS)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return candidates


def _recording_duration_seconds(df: pd.DataFrame) -> float | None:
    """Compute approximate recording duration from TIMESTAMP when possible."""
    if TIME_COLUMN not in df.columns or df.empty:
        return None

    timestamps = df[TIME_COLUMN].dropna()
    if timestamps.empty:
        return None

    numeric_timestamps = pd.to_numeric(timestamps, errors="coerce")
    if numeric_timestamps.notna().sum() >= 2:
        duration = numeric_timestamps.max() - numeric_timestamps.min()
        if pd.notna(duration):
            return float(duration)

    datetime_timestamps = pd.to_datetime(timestamps, errors="coerce")
    if datetime_timestamps.notna().sum() >= 2:
        duration = datetime_timestamps.max() - datetime_timestamps.min()
        if pd.notna(duration):
            return float(duration.total_seconds())

    return None


def _numeric_summary(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, dict[str, float | None]]:
    """Return min/mean/std/max summaries for present numeric columns."""
    summary: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            summary[column] = {
                "min": float(values.min()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "max": float(values.max()),
            }
        else:
            summary[column] = {
                "min": None,
                "mean": None,
                "std": None,
                "max": None,
            }
    return summary


def _value_counts(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, dict[str, int]]:
    """Return value-count dictionaries for present columns."""
    counts: dict[str, dict[str, int]] = {}
    for column in columns:
        if column in df.columns:
            counts[column] = {
                str(value): int(count)
                for value, count in df[column].value_counts(dropna=False).items()
            }
    return counts


def _empty_participant_summary(
    csv_path: Path,
    participant_id: str | None,
    error: str,
) -> dict[str, object]:
    """Return the sparse participant inventory row used for unreadable files."""
    return {
        "participant_id": participant_id,
        "file_path": str(csv_path),
        "n_rows": None,
        "n_columns": None,
        "available_columns": _json_dumps([]),
        "has_expected_schema": False,
        "has_all_expected_columns": False,
        "missing_expected_columns": _json_dumps(EXPECTED_DREAMT_COLUMNS),
        "extra_columns": _json_dumps([]),
        "missing_expected_signal_columns": _json_dumps(EXPECTED_SIGNAL_COLUMNS),
        "recording_duration_seconds": None,
        "label_column": None,
        "label_column_candidates": _json_dumps([]),
        "unique_label_values": _json_dumps([]),
        "label_counts": _json_dumps({}),
        "missing_value_counts_by_signal": _json_dumps({}),
        "missing_value_percentages_by_signal": _json_dumps({}),
        "signal_summary_stats": _json_dumps({}),
        "event_annotation_value_counts": _json_dumps({}),
        "event_annotation_unique_values": _json_dumps({}),
        "error": error,
        **{f"has_{column}": False for column in EXPECTED_SIGNAL_COLUMNS},
        **{f"has_{column}": False for column in EVENT_ANNOTATION_COLUMNS},
    }


def _read_csv_header(file_path: Path) -> list[str]:
    """Read only a CSV header while preserving participant-loader errors."""
    if not file_path.exists():
        raise FileNotFoundError(f"Participant CSV does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Participant CSV path is not a file: {file_path}")

    try:
        return list(pd.read_csv(file_path, nrows=0).columns)
    except Exception as exc:
        raise ValueError(f"Could not read participant CSV {file_path}: {exc}") from exc


def _ordered_count_dict(counts: Counter[str]) -> dict[str, int]:
    """Return count dictionaries in a stable value-count-like order."""
    return {
        key: int(count)
        for key, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def _update_string_counts(counts: Counter[str], values: pd.Series) -> None:
    """Accumulate pandas-style value counts as JSON-safe string keys."""
    for value, count in values.value_counts(dropna=False).items():
        counts[str(value)] += int(count)


def _new_numeric_accumulator() -> dict[str, float | int | None]:
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": None,
        "max": None,
    }


def _update_numeric_accumulator(
    accumulator: dict[str, float | int | None],
    values: pd.Series,
) -> None:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return

    array = numeric_values.to_numpy(dtype=float)
    count = int(array.size)
    accumulator["count"] = int(accumulator["count"]) + count
    accumulator["sum"] = float(accumulator["sum"]) + float(array.sum())
    accumulator["sum_sq"] = float(accumulator["sum_sq"]) + float(
        np.square(array).sum()
    )

    chunk_min = float(array.min())
    chunk_max = float(array.max())
    accumulator["min"] = (
        chunk_min
        if accumulator["min"] is None
        else min(float(accumulator["min"]), chunk_min)
    )
    accumulator["max"] = (
        chunk_max
        if accumulator["max"] is None
        else max(float(accumulator["max"]), chunk_max)
    )


def _finalize_numeric_accumulator(
    accumulator: dict[str, float | int | None],
) -> dict[str, float | None]:
    count = int(accumulator["count"])
    if count == 0:
        return {
            "min": None,
            "mean": None,
            "std": None,
            "max": None,
        }

    total = float(accumulator["sum"])
    mean = total / count
    if count > 1:
        variance = (float(accumulator["sum_sq"]) - (total * total / count)) / (
            count - 1
        )
        std = float(np.sqrt(max(variance, 0.0)))
    else:
        std = float("nan")

    return {
        "min": float(accumulator["min"]),
        "mean": float(mean),
        "std": std,
        "max": float(accumulator["max"]),
    }


def _new_timestamp_accumulator() -> dict[str, object]:
    return {
        "numeric_count": 0,
        "numeric_min": None,
        "numeric_max": None,
        "datetime_count": 0,
        "datetime_min": None,
        "datetime_max": None,
    }


def _update_timestamp_accumulator(
    accumulator: dict[str, object],
    values: pd.Series,
) -> None:
    timestamps = values.dropna()
    if timestamps.empty:
        return

    numeric_values = pd.to_numeric(timestamps, errors="coerce").dropna()
    if not numeric_values.empty:
        numeric_array = numeric_values.to_numpy(dtype=float)
        accumulator["numeric_count"] = (
            int(accumulator["numeric_count"]) + numeric_array.size
        )
        numeric_min = float(numeric_array.min())
        numeric_max = float(numeric_array.max())
        accumulator["numeric_min"] = (
            numeric_min
            if accumulator["numeric_min"] is None
            else min(float(accumulator["numeric_min"]), numeric_min)
        )
        accumulator["numeric_max"] = (
            numeric_max
            if accumulator["numeric_max"] is None
            else max(float(accumulator["numeric_max"]), numeric_max)
        )

    datetime_values = pd.to_datetime(timestamps, errors="coerce").dropna()
    if datetime_values.empty:
        return

    accumulator["datetime_count"] = (
        int(accumulator["datetime_count"]) + len(datetime_values)
    )
    datetime_min = datetime_values.min()
    datetime_max = datetime_values.max()
    accumulator["datetime_min"] = (
        datetime_min
        if accumulator["datetime_min"] is None
        else min(accumulator["datetime_min"], datetime_min)
    )
    accumulator["datetime_max"] = (
        datetime_max
        if accumulator["datetime_max"] is None
        else max(accumulator["datetime_max"], datetime_max)
    )


def _finalize_timestamp_duration(accumulator: dict[str, object]) -> float | None:
    if int(accumulator["numeric_count"]) >= 2:
        duration = float(accumulator["numeric_max"]) - float(
            accumulator["numeric_min"]
        )
        if pd.notna(duration):
            return float(duration)

    if int(accumulator["datetime_count"]) >= 2:
        duration = accumulator["datetime_max"] - accumulator["datetime_min"]
        if pd.notna(duration):
            return float(duration.total_seconds())

    return None


def _iter_inventory_chunks(
    csv_path: Path,
    columns: list[str],
    requested_columns: Iterable[str],
    chunksize: int | None,
) -> Iterable[pd.DataFrame]:
    """Yield only columns needed for inventory summaries."""
    use_columns = [column for column in columns if column in set(requested_columns)]
    read_columns = use_columns if use_columns else columns[:1]
    read_kwargs = {"usecols": read_columns}

    try:
        if chunksize is None:
            yield pd.read_csv(csv_path, **read_kwargs)
        else:
            yield from pd.read_csv(csv_path, chunksize=chunksize, **read_kwargs)
    except Exception as exc:
        raise ValueError(f"Could not read participant CSV {csv_path}: {exc}") from exc


def _target_count_column(label: str) -> str:
    """Return the summary column name for a target sleep-stage label."""
    return f"{TARGET_LABEL_COLUMN_MAP[label]}_count"


def _target_percentage_column(label: str) -> str:
    """Return the percentage column name for a target sleep-stage label."""
    return f"{TARGET_LABEL_COLUMN_MAP[label]}_percentage"


def _coerce_target_label(value: object) -> str | None:
    """Map common DREAMT label values to the three target labels when possible."""
    if pd.isna(value):
        return None

    try:
        from src.preprocessing import map_sleep_stage

        mapped_label = map_sleep_stage(value)
        if mapped_label in TARGET_LABELS:
            return mapped_label
    except ImportError:
        pass

    label_text = str(value).strip()
    if label_text in {"Wake", "Non-REM", "REM"}:
        return label_text
    if label_text == "Non_REM":
        return "Non-REM"
    return None


def _participant_label_counts_from_long_summary(
    label_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | None:
    """Create participant target counts from long-form label summaries."""
    required_columns = {"participant_id", "mapped_label", "count"}
    if not required_columns.issubset(label_summary.columns):
        return None

    summary = label_summary.copy()
    if "scope" in summary.columns:
        summary = summary[summary["scope"] == "participant"]

    summary = summary[summary["participant_id"].notna()].copy()
    summary["participant_id"] = summary["participant_id"].astype(str).str.strip()
    summary["target_label"] = summary["mapped_label"].map(_coerce_target_label)
    summary["count"] = pd.to_numeric(summary["count"], errors="coerce").fillna(0)
    summary = summary[summary["target_label"].notna()]

    if summary.empty:
        return pd.DataFrame(columns=["participant_id"]), "total_epochs"

    participant_counts = (
        summary.pivot_table(
            index="participant_id",
            columns="target_label",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for label in TARGET_LABELS:
        if label not in participant_counts.columns:
            participant_counts[label] = 0
        participant_counts[_target_count_column(label)] = participant_counts[label]
    count_columns = [_target_count_column(label) for label in TARGET_LABELS]
    return participant_counts[["participant_id", *count_columns]], "total_epochs"


def _participant_label_counts_from_wide_summary(
    label_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | None:
    """Create participant target counts from wide participant-level tables."""
    if "participant_id" not in label_summary.columns:
        return None

    count_sources = {
        "Wake": ["Wake_count", "Wake"],
        "Non-REM": ["Non_REM_count", "Non-REM", "Non_REM"],
        "REM": ["REM_count", "REM"],
    }
    available_sources = {
        label: next(
            (column for column in possible_columns if column in label_summary.columns),
            None,
        )
        for label, possible_columns in count_sources.items()
    }
    if not any(available_sources.values()):
        return None

    participant_counts = pd.DataFrame(
        {"participant_id": label_summary["participant_id"].astype(str).str.strip()}
    )
    for label, source_column in available_sources.items():
        target_column = _target_count_column(label)
        if source_column is None:
            participant_counts[target_column] = 0
        else:
            participant_counts[target_column] = pd.to_numeric(
                label_summary[source_column], errors="coerce"
            ).fillna(0)

    total_source = next(
        (
            column
            for column in ["total_rows", "n_rows", "total_epochs"]
            if column in label_summary.columns
        ),
        None,
    )
    if total_source is not None:
        total_column = (
            "total_rows"
            if total_source in {"total_rows", "n_rows"}
            else "total_epochs"
        )
        participant_counts[total_column] = pd.to_numeric(
            label_summary[total_source], errors="coerce"
        ).fillna(0)
    else:
        total_column = "total_epochs"

    return participant_counts, total_column


def _participant_label_counts_from_label_count_json(
    label_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | None:
    """Create participant target counts from participant_summary label_counts JSON."""
    if not {"participant_id", "label_counts"}.issubset(label_summary.columns):
        return None

    rows: list[dict[str, object]] = []
    for _, row in label_summary.iterrows():
        participant_id = row["participant_id"]
        if pd.isna(participant_id):
            continue

        raw_counts = row["label_counts"]
        if isinstance(raw_counts, str):
            try:
                raw_counts = json.loads(raw_counts)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Could not parse label_counts JSON for participant "
                    f"{participant_id}: {exc}"
                ) from exc

        if not isinstance(raw_counts, dict):
            raise ValueError(
                "label_counts values must be dictionaries or JSON objects. "
                f"Found {type(raw_counts).__name__} for participant {participant_id}."
            )

        count_row = {"participant_id": str(participant_id).strip()}
        for label in TARGET_LABELS:
            count_row[_target_count_column(label)] = 0

        for raw_label, count in raw_counts.items():
            target_label = _coerce_target_label(raw_label)
            if target_label is None:
                continue
            count_row[_target_count_column(target_label)] += int(count)

        if "n_rows" in label_summary.columns:
            count_row["total_rows"] = row["n_rows"]
        rows.append(count_row)

    total_column = "total_rows" if "n_rows" in label_summary.columns else "total_epochs"
    return pd.DataFrame(rows), total_column


def _participant_label_counts_from_epoch_rows(
    epoch_or_label_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | None:
    """Create participant target counts from row-level labels when available."""
    if "participant_id" not in epoch_or_label_summary.columns:
        return None

    label_column = next(
        (
            column
            for column in ["mapped_label", "target_label", LABEL_COLUMN, "label"]
            if column in epoch_or_label_summary.columns
        ),
        None,
    )
    if label_column is None:
        return None

    row_labels = epoch_or_label_summary[["participant_id", label_column]].copy()
    row_labels["participant_id"] = row_labels["participant_id"].astype(str).str.strip()
    row_labels["target_label"] = row_labels[label_column].map(_coerce_target_label)
    row_labels = row_labels[row_labels["target_label"].notna()]

    if row_labels.empty:
        return pd.DataFrame(columns=["participant_id"]), "total_rows"

    participant_counts = (
        row_labels.pivot_table(
            index="participant_id",
            columns="target_label",
            values=label_column,
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for label in TARGET_LABELS:
        if label not in participant_counts.columns:
            participant_counts[label] = 0
        participant_counts[_target_count_column(label)] = participant_counts[label]
    count_columns = [_target_count_column(label) for label in TARGET_LABELS]
    return participant_counts[["participant_id", *count_columns]], "total_rows"


def summarize_split_label_distribution(
    epoch_or_label_summary: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize target-label counts and percentages by participant split.

    The summary accepts the existing long-form label mapping output
    (``participant_id``, ``mapped_label``, ``count``), wide participant-level
    count tables, participant inventory rows with ``label_counts`` JSON, or
    row-level data with participant IDs and mapped/raw sleep-stage labels.
    """
    check_no_participant_overlap(split_df)
    split_assignments = split_df[SPLIT_ASSIGNMENT_COLUMNS].copy()
    split_assignments["participant_id"] = (
        split_assignments["participant_id"].astype(str).str.strip()
    )

    count_result = (
        _participant_label_counts_from_long_summary(epoch_or_label_summary)
        or _participant_label_counts_from_wide_summary(epoch_or_label_summary)
        or _participant_label_counts_from_label_count_json(epoch_or_label_summary)
        or _participant_label_counts_from_epoch_rows(epoch_or_label_summary)
    )
    if count_result is None:
        raise ValueError(
            "Could not summarize label distribution. Provide participant-level "
            "target counts, long-form label mapping counts, participant_summary "
            "label_counts, or row-level labels with participant_id."
        )

    participant_counts, total_column = count_result
    participant_counts = participant_counts.copy()
    if "participant_id" in participant_counts.columns:
        participant_counts["participant_id"] = (
            participant_counts["participant_id"].astype(str).str.strip()
        )

    count_columns = [_target_count_column(label) for label in TARGET_LABELS]
    for count_column in count_columns:
        if count_column not in participant_counts.columns:
            participant_counts[count_column] = 0
        participant_counts[count_column] = pd.to_numeric(
            participant_counts[count_column], errors="coerce"
        ).fillna(0)

    if total_column in participant_counts.columns:
        participant_counts[total_column] = pd.to_numeric(
            participant_counts[total_column], errors="coerce"
        ).fillna(0)
    else:
        participant_counts[total_column] = participant_counts[count_columns].sum(axis=1)

    merged = split_assignments.merge(
        participant_counts, on="participant_id", how="left"
    )
    for column in [*count_columns, total_column]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)

    split_order = pd.CategoricalDtype(SPLIT_LABEL_ORDER, ordered=True)
    merged["split"] = merged["split"].astype(split_order)

    summary = (
        merged.groupby("split", observed=False)
        .agg(
            num_participants=("participant_id", "nunique"),
            **{total_column: (total_column, "sum")},
            **{column: (column, "sum") for column in count_columns},
        )
        .reset_index()
    )

    for label in TARGET_LABELS:
        count_column = _target_count_column(label)
        percentage_column = _target_percentage_column(label)
        denominator = summary[total_column].where(summary[total_column] > 0)
        summary[percentage_column] = (
            (summary[count_column] / denominator) * 100
        ).fillna(0)

    ordered_columns = [
        "split",
        "num_participants",
        total_column,
        *count_columns,
        *[_target_percentage_column(label) for label in TARGET_LABELS],
    ]
    return summary[ordered_columns]


def summarize_participant_file(
    file_path: str | Path,
    chunksize: int | None = DEFAULT_INVENTORY_CHUNKSIZE,
) -> dict[str, object]:
    """Create a non-modeling inventory summary for one DREAMT participant CSV.

    Unreadable files return a sparse row with an ``error`` message so the full
    dataset inventory can document corrupted files without stopping early.
    """
    if chunksize is not None and (
        not isinstance(chunksize, int) or chunksize <= 0
    ):
        raise ValueError("chunksize must be a positive integer or None.")

    csv_path = Path(file_path)
    try:
        participant_id = extract_participant_id(csv_path)
    except ValueError:
        participant_id = None

    try:
        columns = _read_csv_header(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        return _empty_participant_summary(csv_path, participant_id, str(exc))

    columns_set = set(columns)
    missing_signal_columns = [
        column for column in EXPECTED_SIGNAL_COLUMNS if column not in columns_set
    ]
    missing_expected_columns = [
        column for column in EXPECTED_DREAMT_COLUMNS if column not in columns_set
    ]
    extra_columns = [
        column for column in columns if column not in EXPECTED_DREAMT_COLUMNS
    ]

    label_match = identify_label_columns(columns)
    label_column = label_match if isinstance(label_match, str) else None
    label_candidates = (
        label_match
        if isinstance(label_match, list)
        else ([label_match] if label_match is not None else [])
    )

    present_signal_columns = [
        column for column in EXPECTED_SIGNAL_COLUMNS if column in columns_set
    ]
    present_event_columns = [
        column for column in EVENT_ANNOTATION_COLUMNS if column in columns_set
    ]
    requested_columns = {
        TIME_COLUMN,
        *present_signal_columns,
        *present_event_columns,
    }
    if label_column is not None:
        requested_columns.add(label_column)

    n_rows = 0
    label_counter: Counter[str] = Counter()
    missing_counts = {column: 0 for column in present_signal_columns}
    signal_accumulators = {
        column: _new_numeric_accumulator() for column in present_signal_columns
    }
    event_counters = {column: Counter() for column in present_event_columns}
    timestamp_accumulator = _new_timestamp_accumulator()

    try:
        for chunk in _iter_inventory_chunks(
            csv_path,
            columns=columns,
            requested_columns=requested_columns,
            chunksize=chunksize,
        ):
            n_rows += len(chunk)

            if label_column is not None and label_column in chunk.columns:
                _update_string_counts(label_counter, chunk[label_column])

            if TIME_COLUMN in chunk.columns:
                _update_timestamp_accumulator(
                    timestamp_accumulator,
                    chunk[TIME_COLUMN],
                )

            for column in present_signal_columns:
                if column not in chunk.columns:
                    continue
                missing_counts[column] += int(chunk[column].isna().sum())
                _update_numeric_accumulator(
                    signal_accumulators[column],
                    chunk[column],
                )

            for column in present_event_columns:
                if column in chunk.columns:
                    _update_string_counts(event_counters[column], chunk[column])
    except ValueError as exc:
        return _empty_participant_summary(csv_path, participant_id, str(exc))

    label_counts = _ordered_count_dict(label_counter)
    unique_label_values = list(label_counts)
    missing_percentages = {
        column: (float(missing_counts[column] / n_rows * 100) if n_rows else None)
        for column in present_signal_columns
    }

    event_counts = {
        column: _ordered_count_dict(event_counters[column])
        for column in present_event_columns
    }
    event_unique_values = {
        column: list(counts) for column, counts in event_counts.items()
    }

    summary: dict[str, object] = {
        "participant_id": participant_id,
        "file_path": str(csv_path),
        "n_rows": int(n_rows),
        "n_columns": int(len(columns)),
        "available_columns": _json_dumps(columns),
        "has_expected_schema": columns == EXPECTED_DREAMT_COLUMNS,
        "has_all_expected_columns": not missing_expected_columns,
        "missing_expected_columns": _json_dumps(missing_expected_columns),
        "extra_columns": _json_dumps(extra_columns),
        "missing_expected_signal_columns": _json_dumps(missing_signal_columns),
        "recording_duration_seconds": _finalize_timestamp_duration(
            timestamp_accumulator
        ),
        "label_column": label_column,
        "label_column_candidates": _json_dumps(label_candidates),
        "unique_label_values": _json_dumps(unique_label_values),
        "label_counts": _json_dumps(label_counts),
        "missing_value_counts_by_signal": _json_dumps(missing_counts),
        "missing_value_percentages_by_signal": _json_dumps(missing_percentages),
        "signal_summary_stats": _json_dumps(
            {
                column: _finalize_numeric_accumulator(signal_accumulators[column])
                for column in present_signal_columns
            }
        ),
        "event_annotation_value_counts": _json_dumps(event_counts),
        "event_annotation_unique_values": _json_dumps(event_unique_values),
        "error": None,
    }

    for column in EXPECTED_SIGNAL_COLUMNS:
        summary[f"has_{column}"] = column in columns_set
    for column in EVENT_ANNOTATION_COLUMNS:
        summary[f"has_{column}"] = column in columns_set

    return summary


def summarize_dataset(
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    output_path: str | Path = DEFAULT_PARTICIPANT_SUMMARY_PATH,
    pattern: str = DEFAULT_PARTICIPANT_PATTERN,
    chunksize: int | None = DEFAULT_INVENTORY_CHUNKSIZE,
) -> pd.DataFrame:
    """Summarize all participant CSV files and save the summary CSV.

    Returns a DataFrame with one row per participant. ``data/interim/`` is
    created when needed.
    """
    participant_files = list_participant_csvs(raw_data_dir, pattern=pattern)
    print(f"Found {len(participant_files)} participant CSV file(s) in {raw_data_dir}.")

    summaries = [
        summarize_participant_file(path, chunksize=chunksize)
        for path in participant_files
    ]
    summary_df = pd.DataFrame(summaries, columns=PARTICIPANT_SUMMARY_COLUMNS)

    output_csv = Path(output_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved participant summary to {output_csv}.")

    return summary_df
