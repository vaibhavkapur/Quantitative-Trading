"""Chronological train/calibration/test splitting with explicit future-label leakage guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from volatility_forecasting.config import SplitConfig
from volatility_forecasting.errors import (
    ConfigurationError,
    FutureLeakageError,
    UnsupportedOptionError,
)


@dataclass(frozen=True)
class ChronologicalSplit:
    split_id: int
    horizon_days: int
    embargo_steps: int
    train_start_ts: str
    train_end_ts: str
    inner_train_end_ts: str
    cal_start_ts: str
    cal_end_ts: str
    test_start_ts: str
    test_end_ts: str
    n_train: int
    n_inner_train: int
    n_cal: int
    n_test: int
    train_indices: tuple[int, ...]
    inner_train_indices: tuple[int, ...]
    cal_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    def summary_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("train_indices")
        d.pop("inner_train_indices")
        d.pop("cal_indices")
        d.pop("test_indices")
        return d


def assert_no_future_leakage(
    df: pl.DataFrame,
    train_indices: list[int] | tuple[int, ...],
    test_indices: list[int] | tuple[int, ...],
    horizon_days: int,
) -> None:
    """Strictly verify zero future label or timestamp leakage between train and test windows.

    Checks:
      1. Neither train nor test index set is empty, and they are disjoint.
      2. Every training origin timestamp is strictly earlier than the earliest test timestamp.
      3. Every training row's forward label end timestamp (`label_end_ts_h{horizon}`) is
         strictly earlier than the earliest test timestamp (no target window overlap).
    """
    if not train_indices or not test_indices:
        raise FutureLeakageError("Train and test index sets must both be non-empty.")

    overlap = set(train_indices).intersection(test_indices)
    if overlap:
        raise FutureLeakageError(
            f"Train and test index sets overlap on {len(overlap)} rows "
            f"(sample indices: {sorted(overlap)[:5]})."
        )

    train_sub = df[list(train_indices)]
    test_sub = df[list(test_indices)]

    max_train_ts = train_sub["timestamp"].max()
    min_test_ts = test_sub["timestamp"].min()
    if max_train_ts is None or min_test_ts is None:
        raise FutureLeakageError("Null timestamps detected while validating leakage bounds.")
    if max_train_ts >= min_test_ts:
        raise FutureLeakageError(
            f"Chronological ordering violation: max train timestamp ({max_train_ts}) "
            f">= min test timestamp ({min_test_ts})."
        )

    label_end_col = f"label_end_ts_h{horizon_days}"
    if label_end_col not in df.columns:
        raise FutureLeakageError(
            f"Cannot verify future label leakage: missing column '{label_end_col}'."
        )
    max_train_label_end = train_sub[label_end_col].max()
    if max_train_label_end is None:
        raise FutureLeakageError(
            f"Training rows have null '{label_end_col}' values; labels must be well-defined."
        )
    if max_train_label_end >= min_test_ts:
        raise FutureLeakageError(
            f"Future label leakage detected for horizon={horizon_days}: "
            f"training label window extends to {max_train_label_end}, "
            f"overlapping test start {min_test_ts}. Increase embargo_steps >= {horizon_days}."
        )


def generate_chronological_splits(
    df: pl.DataFrame,
    split_cfg: SplitConfig,
    horizon_days: int,
) -> list[ChronologicalSplit]:
    """Generate expanding or rolling chronological splits with purge/embargo gap.

    Splitting operates on unique sorted timestamps so multi-asset panels never leak
    across assets on the same date.
    """
    if split_cfg.scheme not in {"expanding", "rolling"}:
        raise UnsupportedOptionError(f"Unsupported split scheme '{split_cfg.scheme}'.")

    target_col = f"target_rv_h{horizon_days}"
    label_end_col = f"label_end_ts_h{horizon_days}"
    if target_col not in df.columns or label_end_col not in df.columns:
        raise ConfigurationError(
            f"Dataset missing horizon={horizon_days} target columns ('{target_col}', '{label_end_col}')."
        )

    # Unique timestamps that have non-null targets
    valid_mask = df[target_col].is_not_null() & df[label_end_col].is_not_null()
    valid_df = df.with_row_index("_pos").filter(valid_mask)
    unique_ts = valid_df["timestamp"].unique().sort().to_list()
    n_ts = len(unique_ts)

    embargo = (
        split_cfg.embargo_steps
        if split_cfg.embargo_steps is not None
        else max(horizon_days, 1)
    )
    if embargo < horizon_days:
        raise FutureLeakageError(
            f"Configured embargo_steps ({embargo}) < horizon_days ({horizon_days}); "
            "this would allow training label windows to leak into the test set."
        )

    required_min = split_cfg.min_train_size + embargo + split_cfg.test_size
    if n_ts < required_min:
        raise ConfigurationError(
            f"Insufficient valid timestamps ({n_ts}) for min_train_size={split_cfg.min_train_size}, "
            f"embargo={embargo}, test_size={split_cfg.test_size} (need >= {required_min})."
        )

    # Compute split anchor points evenly across remaining timeline
    max_test_start = n_ts - split_cfg.test_size
    first_test_start = split_cfg.min_train_size + embargo
    if split_cfg.n_splits == 1:
        test_starts = [first_test_start]
    else:
        available_span = max_test_start - first_test_start
        if available_span <= 0:
            test_starts = [first_test_start]
        else:
            raw_starts = np.linspace(first_test_start, max_test_start, num=split_cfg.n_splits)
            test_starts = sorted({int(round(x)) for x in raw_starts})

    # Map each timestamp to its row indices in the original df
    ts_to_indices: dict[object, list[int]] = {}
    for row in valid_df.select(["_pos", "timestamp"]).iter_rows(named=True):
        ts_to_indices.setdefault(row["timestamp"], []).append(int(row["_pos"]))

    splits: list[ChronologicalSplit] = []
    for split_idx, test_start_pos in enumerate(test_starts, start=1):
        train_end_pos = test_start_pos - embargo  # exclusive end in unique_ts
        if split_cfg.scheme == "expanding":
            train_start_pos = 0
        else:
            train_start_pos = max(0, train_end_pos - split_cfg.min_train_size)

        test_end_pos = min(n_ts, test_start_pos + split_cfg.test_size)

        train_ts_slice = unique_ts[train_start_pos:train_end_pos]
        test_ts_slice = unique_ts[test_start_pos:test_end_pos]

        # Split training window into inner-train + embargo + calibration tail for conformal intervals
        n_train_ts = len(train_ts_slice)
        cal_len = max(5, int(round(n_train_ts * split_cfg.calibration_fraction)))
        if n_train_ts > cal_len + embargo + 10:
            cal_start_rel = n_train_ts - cal_len
            inner_end_rel = cal_start_rel - embargo
            inner_ts_slice = train_ts_slice[:inner_end_rel]
            cal_ts_slice = train_ts_slice[cal_start_rel:]
        else:
            # Fallback when training window is very compact: use trailing 20% without shrinking inner below 5
            cal_len = max(2, n_train_ts // 5)
            inner_end_rel = max(5, n_train_ts - cal_len)
            inner_ts_slice = train_ts_slice[:inner_end_rel]
            cal_ts_slice = train_ts_slice[inner_end_rel:]

        train_indices = tuple(i for ts in train_ts_slice for i in ts_to_indices[ts])
        inner_train_indices = tuple(i for ts in inner_ts_slice for i in ts_to_indices[ts])
        cal_indices = tuple(i for ts in cal_ts_slice for i in ts_to_indices[ts])
        test_indices = tuple(i for ts in test_ts_slice for i in ts_to_indices[ts])

        # Enforce zero future leakage invariant on every split
        assert_no_future_leakage(df, train_indices, test_indices, horizon_days=horizon_days)

        splits.append(
            ChronologicalSplit(
                split_id=split_idx,
                horizon_days=horizon_days,
                embargo_steps=embargo,
                train_start_ts=str(train_ts_slice[0]),
                train_end_ts=str(train_ts_slice[-1]),
                inner_train_end_ts=str(inner_ts_slice[-1]),
                cal_start_ts=str(cal_ts_slice[0]),
                cal_end_ts=str(cal_ts_slice[-1]),
                test_start_ts=str(test_ts_slice[0]),
                test_end_ts=str(test_ts_slice[-1]),
                n_train=len(train_indices),
                n_inner_train=len(inner_train_indices),
                n_cal=len(cal_indices),
                n_test=len(test_indices),
                train_indices=train_indices,
                inner_train_indices=inner_train_indices,
                cal_indices=cal_indices,
                test_indices=test_indices,
            )
        )

    return splits
