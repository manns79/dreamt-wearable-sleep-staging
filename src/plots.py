"""Reusable EDA summaries and plots for DREAMT sleep staging analysis.

The functions in this module accept explicit DataFrames and paths so notebooks
can stay focused on interpretation while keeping leakage-sensitive filtering
visible at the call site.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from src.data import EXPECTED_SIGNAL_COLUMNS, load_participant_csv
from src.features import ACC_MAG_COLUMN, compute_acc_magnitude
from src.preprocessing import DEFAULT_EPOCH_LENGTH_SECONDS, TARGET_SLEEP_STAGE_LABELS

MISSINGNESS_PREFIX = "missingness_"
EPOCH_SUMMARY_STATS = ("mean", "std", "median", "iqr")


def save_figure(fig, path: str | Path | None = None, **savefig_kwargs) -> None:
    """Save a matplotlib figure when a path is provided."""
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150, **savefig_kwargs)


def add_acc_magnitude(
    df: pd.DataFrame,
    x_col: str = "ACC_X",
    y_col: str = "ACC_Y",
    z_col: str = "ACC_Z",
    output_col: str = ACC_MAG_COLUMN,
) -> pd.DataFrame:
    """Return a copy of ``df`` with accelerometer magnitude added."""
    output = df.copy()
    output[output_col] = compute_acc_magnitude(
        output,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
    )
    return output


def valid_training_epochs(
    epoch_index: pd.DataFrame,
    split: str = "train",
    require_valid: bool = True,
) -> pd.DataFrame:
    """Filter an epoch index to one split, optionally valid epochs only."""
    required_columns = {"participant_id", "split", "mapped_label"}
    missing_columns = sorted(required_columns - set(epoch_index.columns))
    if missing_columns:
        raise ValueError(f"epoch_index is missing column(s): {missing_columns}")

    epochs = epoch_index[epoch_index["split"].astype(str) == split].copy()
    if require_valid:
        if "is_valid_epoch" not in epochs.columns:
            raise ValueError("epoch_index is missing column: is_valid_epoch")
        epochs = epochs[epochs["is_valid_epoch"].astype(bool)].copy()

    labels = set(TARGET_SLEEP_STAGE_LABELS)
    epochs = epochs[epochs["mapped_label"].isin(labels)].copy()
    return epochs.reset_index(drop=True)


def summarize_class_balance(
    epochs: pd.DataFrame,
    label_col: str = "mapped_label",
    labels: Iterable[str] = TARGET_SLEEP_STAGE_LABELS,
) -> pd.DataFrame:
    """Count and percent epochs in each sleep-stage class."""
    labels = list(labels)
    counts = epochs[label_col].value_counts().reindex(labels, fill_value=0)
    total = int(counts.sum())
    percentages = (counts / total * 100) if total else counts.astype(float)

    return pd.DataFrame(
        {
            "mapped_label": labels,
            "n_epochs": counts.astype(int).to_numpy(),
            "percentage": percentages.fillna(0).to_numpy(),
        }
    )


def summarize_participant_class_distribution(
    epochs: pd.DataFrame,
    participant_col: str = "participant_id",
    label_col: str = "mapped_label",
    labels: Iterable[str] = TARGET_SLEEP_STAGE_LABELS,
) -> pd.DataFrame:
    """Summarize per-participant class counts and percentages."""
    labels = list(labels)
    counts = (
        epochs.groupby([participant_col, label_col])
        .size()
        .unstack(label_col, fill_value=0)
        .reindex(columns=labels, fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    counts["total_epochs"] = counts[labels].sum(axis=1)

    percentages = counts[[participant_col, "total_epochs"]].copy()
    for label in labels:
        percentages[f"{label}_percentage"] = np.where(
            counts["total_epochs"] > 0,
            counts[label] / counts["total_epochs"] * 100,
            0,
        )
        counts[f"{label}_count"] = counts[label].astype(int)

    ordered_columns = [
        participant_col,
        "total_epochs",
        *[f"{label}_count" for label in labels],
        *[f"{label}_percentage" for label in labels],
    ]
    return counts.merge(
        percentages[
            [participant_col, *[f"{label}_percentage" for label in labels]]
        ],
        on=participant_col,
        suffixes=("", "_duplicate"),
    )[ordered_columns]


def _missingness_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column.startswith(MISSINGNESS_PREFIX)
        and column.removeprefix(MISSINGNESS_PREFIX) in EXPECTED_SIGNAL_COLUMNS
    ]


def summarize_missingness_by_signal(epochs: pd.DataFrame) -> pd.DataFrame:
    """Summarize epoch-level missingness fractions by signal."""
    rows: list[dict[str, float | str]] = []
    for column in _missingness_columns(epochs):
        values = pd.to_numeric(epochs[column], errors="coerce")
        rows.append(
            {
                "signal": column.removeprefix(MISSINGNESS_PREFIX),
                "mean_missingness": float(values.mean()) if values.notna().any() else 0,
                "median_missingness": (
                    float(values.median()) if values.notna().any() else 0
                ),
                "p95_missingness": (
                    float(values.quantile(0.95)) if values.notna().any() else 0
                ),
                "max_missingness": float(values.max()) if values.notna().any() else 0,
            }
        )
    return pd.DataFrame(rows)


def summarize_missingness_by_group(
    epochs: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Return mean missingness by a grouping column and signal."""
    if group_col not in epochs.columns:
        raise ValueError(f"epochs is missing grouping column: {group_col}")

    rows: list[dict[str, float | str]] = []
    for group_value, group_df in epochs.groupby(group_col, dropna=False):
        for column in _missingness_columns(group_df):
            values = pd.to_numeric(group_df[column], errors="coerce")
            rows.append(
                {
                    group_col: group_value,
                    "signal": column.removeprefix(MISSINGNESS_PREFIX),
                    "mean_missingness": (
                        float(values.mean()) if values.notna().any() else 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_raw_epoch_signals(
    raw_epoch: pd.DataFrame,
    signal_columns: Iterable[str] = EXPECTED_SIGNAL_COLUMNS,
    include_acc_magnitude: bool = True,
) -> dict[str, float]:
    """Compute simple summary statistics for one raw epoch slice."""
    epoch = raw_epoch.copy()
    signals = list(signal_columns)
    if include_acc_magnitude and {"ACC_X", "ACC_Y", "ACC_Z"}.issubset(epoch.columns):
        epoch = add_acc_magnitude(epoch)
        signals = [*signals, ACC_MAG_COLUMN]

    summary: dict[str, float] = {}
    for signal in signals:
        if signal not in epoch.columns:
            continue
        values = pd.to_numeric(epoch[signal], errors="coerce").dropna()
        if values.empty:
            for stat in EPOCH_SUMMARY_STATS:
                summary[f"{signal}_{stat}"] = np.nan
            continue
        quantiles = values.quantile([0.25, 0.75])
        summary[f"{signal}_mean"] = float(values.mean())
        summary[f"{signal}_std"] = float(values.std(ddof=1))
        summary[f"{signal}_median"] = float(values.median())
        summary[f"{signal}_iqr"] = float(quantiles.loc[0.75] - quantiles.loc[0.25])
    return summary


def _raw_participant_path(raw_data_dir: str | Path, participant_id: str) -> Path:
    """Return the expected raw CSV path for a DREAMT participant."""
    raw_dir = Path(raw_data_dir)
    expected_path = raw_dir / f"{participant_id}_whole_df.csv"
    if expected_path.exists():
        return expected_path

    matches = sorted(raw_dir.glob(f"{participant_id}*_whole_df.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Raw CSV for participant {participant_id} was not found under {raw_dir}."
    )


def select_representative_epochs(
    epochs: pd.DataFrame,
    n_per_stage: int = 1,
    random_state: int = 42,
    label_col: str = "mapped_label",
) -> pd.DataFrame:
    """Select a small deterministic sample of epochs from each stage."""
    if n_per_stage <= 0:
        raise ValueError("n_per_stage must be positive.")
    required_columns = {label_col, "participant_id", "epoch_id"}
    missing_columns = sorted(required_columns - set(epochs.columns))
    if missing_columns:
        raise ValueError(f"epochs is missing column(s): {missing_columns}")

    selected: list[pd.DataFrame] = []
    for label in TARGET_SLEEP_STAGE_LABELS:
        group_df = epochs[epochs[label_col] == label]
        if group_df.empty:
            continue
        selected.append(
            group_df.sample(
                n=min(n_per_stage, len(group_df)),
                random_state=random_state,
            )
        )
    if not selected:
        return epochs.iloc[0:0].copy()
    return pd.concat(selected).sort_values([label_col, "participant_id", "epoch_id"])


def collect_epoch_signal_summaries(
    epochs: pd.DataFrame,
    raw_data_dir: str | Path,
    max_epochs_per_stage: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute raw-signal summaries for indexed epochs one participant at a time."""
    required_columns = {
        "participant_id",
        "epoch_id",
        "start_row",
        "end_row",
        "mapped_label",
    }
    missing_columns = sorted(required_columns - set(epochs.columns))
    if missing_columns:
        raise ValueError(f"epochs is missing column(s): {missing_columns}")

    analysis_epochs = epochs.copy()
    if max_epochs_per_stage is not None:
        sampled_epochs: list[pd.DataFrame] = []
        for label in TARGET_SLEEP_STAGE_LABELS:
            label_epochs = analysis_epochs[analysis_epochs["mapped_label"] == label]
            if label_epochs.empty:
                continue
            sampled_epochs.append(
                label_epochs.sample(
                    n=min(max_epochs_per_stage, len(label_epochs)),
                    random_state=random_state,
                )
            )
        analysis_epochs = (
            pd.concat(sampled_epochs, ignore_index=True)
            if sampled_epochs
            else analysis_epochs.iloc[0:0].copy()
        )

    rows: list[dict[str, object]] = []
    for participant_id, participant_epochs in analysis_epochs.groupby("participant_id"):
        raw_path = _raw_participant_path(raw_data_dir, participant_id)
        raw_df = load_participant_csv(raw_path, usecols=EXPECTED_SIGNAL_COLUMNS)
        for _, epoch_row in participant_epochs.iterrows():
            start_row = int(epoch_row["start_row"])
            end_row = int(epoch_row["end_row"])
            raw_epoch = raw_df.iloc[start_row:end_row]
            rows.append(
                {
                    "participant_id": participant_id,
                    "epoch_id": int(epoch_row["epoch_id"]),
                    "mapped_label": epoch_row["mapped_label"],
                    **summarize_raw_epoch_signals(raw_epoch),
                }
            )

    return pd.DataFrame(rows)


def transition_matrices(
    epochs: pd.DataFrame,
    participant_col: str = "participant_id",
    label_col: str = "mapped_label",
    epoch_id_col: str = "epoch_id",
    labels: Iterable[str] = TARGET_SLEEP_STAGE_LABELS,
    require_consecutive_epoch_ids: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute within-participant sleep-stage transition counts and probabilities."""
    labels = list(labels)
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=int)

    required_columns = {participant_col, label_col}
    if require_consecutive_epoch_ids:
        required_columns.add(epoch_id_col)
    missing_columns = sorted(required_columns - set(epochs.columns))
    if missing_columns:
        raise ValueError(f"epochs is missing column(s): {missing_columns}")

    sort_columns = [participant_col]
    if epoch_id_col in epochs.columns:
        sort_columns.append(epoch_id_col)

    for _, group_df in epochs.sort_values(sort_columns).groupby(participant_col):
        group_df = group_df[group_df[label_col].isin(labels)].copy()
        if len(group_df) < 2:
            continue
        previous = group_df.shift(1)
        previous_epoch_id = (
            previous[epoch_id_col] if epoch_id_col in group_df else np.nan
        )
        transitions = group_df.assign(
            previous_label=previous[label_col],
            previous_epoch_id=previous_epoch_id,
        )
        transitions = transitions[transitions["previous_label"].isin(labels)]
        if require_consecutive_epoch_ids:
            transitions = transitions[
                pd.to_numeric(transitions[epoch_id_col], errors="coerce")
                - pd.to_numeric(transitions["previous_epoch_id"], errors="coerce")
                == 1
            ]
        for _, row in transitions.iterrows():
            counts.loc[row["previous_label"], row[label_col]] += 1

    probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return counts, probabilities


def plot_class_balance(
    class_balance: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """Plot class counts with percentage labels."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=class_balance, x="mapped_label", y="n_epochs", ax=ax)
    ax.set_xlabel("Sleep stage")
    ax.set_ylabel("Training epochs")
    ax.set_title("Training Set Class Balance")
    for container in ax.containers:
        labels = [f"{percentage:.1f}%" for percentage in class_balance["percentage"]]
        ax.bar_label(container, labels=labels)
    save_figure(fig, save_path)
    return fig, ax


def plot_participant_class_distribution(
    participant_distribution: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """Plot participant-level class percentages as stacked bars."""
    import matplotlib.pyplot as plt

    labels = list(TARGET_SLEEP_STAGE_LABELS)
    percentage_columns = [f"{label}_percentage" for label in labels]
    plot_df = participant_distribution.sort_values("total_epochs").reset_index(
        drop=True
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(plot_df))
    for label, column in zip(labels, percentage_columns, strict=True):
        values = plot_df[column].to_numpy()
        ax.bar(plot_df["participant_id"], values, bottom=bottom, label=label)
        bottom += values
    ax.set_xlabel("Training participant")
    ax.set_ylabel("Epoch percentage")
    ax.set_title("Participant-Level Training Class Distribution")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(title="Sleep stage", loc="upper right")
    save_figure(fig, save_path)
    return fig, ax


def plot_missingness_by_signal(
    missingness_by_signal: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """Plot mean epoch missingness by signal."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=missingness_by_signal,
        x="signal",
        y="mean_missingness",
        ax=ax,
    )
    ax.set_xlabel("Signal")
    ax.set_ylabel("Mean missingness fraction")
    ax.set_title("Training Epoch Missingness By Signal")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, save_path)
    return fig, ax


def plot_signal_summary_by_stage(
    summary_df: pd.DataFrame,
    value_col: str,
    signal_col: str = "signal",
    stage_col: str = "mapped_label",
    save_path: str | Path | None = None,
):
    """Plot epoch-level signal summary distributions by sleep stage."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=summary_df,
        x=signal_col,
        y=value_col,
        hue=stage_col,
        hue_order=list(TARGET_SLEEP_STAGE_LABELS),
        showfliers=False,
        ax=ax,
    )
    ax.set_xlabel("Signal")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.set_title(f"Training Epoch {value_col.replace('_', ' ').title()} By Stage")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Sleep stage")
    save_figure(fig, save_path)
    return fig, ax


def plot_raw_epoch_channels(
    epoch_df: pd.DataFrame,
    channels: Iterable[str],
    title: str,
    save_path: str | Path | None = None,
):
    """Plot selected raw channels for one epoch."""
    import matplotlib.pyplot as plt

    plot_df = epoch_df.copy()
    if ACC_MAG_COLUMN in channels and ACC_MAG_COLUMN not in plot_df.columns:
        plot_df = add_acc_magnitude(plot_df)

    channels = [channel for channel in channels if channel in plot_df.columns]
    if not channels:
        raise ValueError("No requested raw epoch channels are present in epoch_df.")

    fig, axes = plt.subplots(
        len(channels),
        1,
        figsize=(11, max(2, len(channels) * 1.5)),
        sharex=True,
    )
    if len(channels) == 1:
        axes = [axes]

    x = np.arange(len(plot_df))
    for ax, channel in zip(axes, channels, strict=True):
        ax.plot(x, pd.to_numeric(plot_df[channel], errors="coerce"), linewidth=0.8)
        ax.set_ylabel(channel)
    axes[-1].set_xlabel("Sample within 30-second epoch")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig, axes


def plot_transition_matrix(
    matrix: pd.DataFrame,
    title: str,
    fmt: str = "d",
    save_path: str | Path | None = None,
):
    """Plot a transition matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Next stage")
    ax.set_ylabel("Previous stage")
    ax.set_title(title)
    save_figure(fig, save_path)
    return fig, ax


def plot_hypnogram(
    epochs: pd.DataFrame,
    participant_id: str,
    save_path: str | Path | None = None,
    epoch_length_seconds: int = DEFAULT_EPOCH_LENGTH_SECONDS,
):
    """Plot sleep stage over elapsed hours for one participant."""
    import matplotlib.pyplot as plt

    stage_to_y = {"Wake": 2, "REM": 1, "Non-REM": 0}
    participant_epochs = (
        epochs[epochs["participant_id"] == participant_id]
        .sort_values("epoch_id")
        .copy()
    )
    participant_epochs["stage_y"] = participant_epochs["mapped_label"].map(stage_to_y)
    first_epoch_id = participant_epochs["epoch_id"].min()
    elapsed_hours = (
        (participant_epochs["epoch_id"] - first_epoch_id)
        * epoch_length_seconds
        / 3600
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(
        elapsed_hours,
        participant_epochs["stage_y"],
        where="post",
        linewidth=1.5,
    )
    ax.set_yticks([0, 1, 2], ["Non-REM", "REM", "Wake"])
    ax.set_xlabel("Hours since first valid epoch")
    ax.set_ylabel("Sleep stage")
    ax.set_title(f"Hypnogram-Like Stage Sequence: {participant_id}")
    save_figure(fig, save_path)
    return fig, ax
