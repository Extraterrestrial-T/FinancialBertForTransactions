"""Leakage-safe tabular features for downstream account-level baselines.

These features deliberately use only rows at or before each task row's
``cutoff_time``.  They provide a serious non-neural comparator for the
pretrained account embedding, rather than a straw-man baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TabularFeatureSchema:
    """Train-fitted feature columns and numeric imputation values.

    Dynamic category-share columns are chosen solely from training examples.
    At validation/test time unseen categories are ignored and absent trained
    categories receive zero share.  Numeric missing values use train medians.
    """

    feature_names: tuple[str, ...]
    fill_values: dict[str, float]

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Align arbitrary split features to this train-only schema."""
        if not features.index.is_unique:
            raise ValueError("feature rows must have unique sample IDs")
        aligned = pd.DataFrame(0.0, index=features.index, columns=self.feature_names)
        for name in self.feature_names:
            if name in features:
                aligned[name] = pd.to_numeric(features[name], errors="coerce")
        for name, fill_value in self.fill_values.items():
            aligned[name] = aligned[name].fillna(fill_value)
        if aligned.isna().any().any():
            raise ValueError("tabular feature transformation left missing values")
        return aligned.astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save this train-fitted preprocessing contract as JSON."""
        Path(path).write_text(
            json.dumps(
                {"feature_names": list(self.feature_names), "fill_values": self.fill_values},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TabularFeatureSchema":
        """Load a schema created by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(str(name) for name in payload["feature_names"]),
            fill_values={str(name): float(value) for name, value in payload["fill_values"].items()},
        )


def build_account_history_features(
    events: pd.DataFrame,
    profiles: pd.DataFrame,
    task_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build one tabular feature row for every labelled account cutoff.

    Args:
        events: Processed transaction rows for the accounts in this split.
        profiles: Processed account-profile rows for the same split.
        task_table: Requires ``sample_id``, ``account_id``, and ``cutoff_time``.

    Returns:
        Float feature matrix indexed by ``sample_id``.  It intentionally
        excludes target, split, and all future-window quantities.
    """
    required_event_columns = {
        "account_id",
        "trans_id",
        "trans_date",
        "amount",
        "balance",
        "trans_type",
        "operation",
        "category",
    }
    required_profile_columns = {"account_id", "create_date"}
    required_task_columns = {"sample_id", "account_id", "cutoff_time"}
    if missing := required_event_columns.difference(events.columns):
        raise ValueError(f"events is missing required columns: {sorted(missing)}")
    if missing := required_profile_columns.difference(profiles.columns):
        raise ValueError(f"profiles is missing required columns: {sorted(missing)}")
    if missing := required_task_columns.difference(task_table.columns):
        raise ValueError(f"task_table is missing required columns: {sorted(missing)}")
    if task_table["sample_id"].duplicated().any():
        raise ValueError("task_table sample_id values must be unique")

    prepared_events = events.copy()
    prepared_events["account_id"] = prepared_events["account_id"].astype(int)
    prepared_events["trans_date"] = pd.to_datetime(prepared_events["trans_date"])
    prepared_events = prepared_events.sort_values(["account_id", "trans_date", "trans_id"])
    events_by_account = {
        int(account_id): group.reset_index(drop=True)
        for account_id, group in prepared_events.groupby("account_id", sort=False)
    }
    category_values = {
        prefix: tuple(sorted({_normalise_category_value(value) for value in prepared_events[column]}))
        for prefix, column in (
            ("trans_type", "trans_type"),
            ("operation", "operation"),
            ("category", "category"),
        )
    }

    profile_dates = profiles[["account_id", "create_date"]].copy()
    profile_dates["account_id"] = profile_dates["account_id"].astype(int)
    profile_dates["create_date"] = pd.to_datetime(profile_dates["create_date"])
    if profile_dates["account_id"].duplicated().any():
        raise ValueError("profiles must have exactly one row per account")
    created_by_account = profile_dates.set_index("account_id")["create_date"].to_dict()

    prepared_tasks = task_table[["sample_id", "account_id", "cutoff_time"]].copy()
    prepared_tasks["account_id"] = prepared_tasks["account_id"].astype(int)
    prepared_tasks["cutoff_time"] = pd.to_datetime(prepared_tasks["cutoff_time"])
    feature_frames: list[pd.DataFrame] = []
    for account_id, account_tasks in prepared_tasks.groupby("account_id", sort=False):
        try:
            account_events = events_by_account[int(account_id)]
            create_date = pd.Timestamp(created_by_account[account_id])
        except KeyError as error:
            raise ValueError(f"task account {account_id} is absent from events or profiles") from error
        if account_events.empty:
            raise ValueError(f"task account {account_id} is absent from events or profiles")
        feature_frames.append(
            _features_for_account_cutoffs(
                account_events,
                account_tasks.sort_values("cutoff_time"),
                create_date,
                category_values,
            )
        )

    return (
        pd.concat(feature_frames, ignore_index=True)
        .set_index("sample_id")
        .sort_index()
        .astype(np.float32)
    )


def fit_tabular_feature_schema(train_features: pd.DataFrame) -> TabularFeatureSchema:
    """Fit feature names and imputation values using training rows only."""
    if train_features.empty:
        raise ValueError("cannot fit a tabular feature schema on no training rows")
    if not train_features.index.is_unique:
        raise ValueError("training feature rows must have unique sample IDs")
    numeric = train_features.apply(pd.to_numeric, errors="coerce")
    feature_names = tuple(
        sorted(
            str(name)
            for name in numeric.columns
            # Dynamic category columns that never occur in a train snapshot
            # are not part of the train-fitted feature vocabulary.
            if not ("_share::" in str(name) and numeric[name].fillna(0.0).eq(0.0).all())
        )
    )
    fill_values: dict[str, float] = {}
    for name in feature_names:
        values = numeric[name].replace([np.inf, -np.inf], np.nan)
        median = values.median(skipna=True)
        fill_values[name] = 0.0 if pd.isna(median) else float(median)
    return TabularFeatureSchema(feature_names=feature_names, fill_values=fill_values)


def _features_for_account_cutoffs(
    events: pd.DataFrame,
    task_rows: pd.DataFrame,
    create_date: pd.Timestamp,
    category_values: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Vectorise feature extraction across every cutoff for one account."""
    event_times = pd.DatetimeIndex(events["trans_date"])
    cutoff_times = pd.DatetimeIndex(task_rows["cutoff_time"])
    history_end = event_times.searchsorted(cutoff_times, side="right")
    if (history_end == 0).any():
        invalid = task_rows.iloc[np.flatnonzero(history_end == 0)[0]]["sample_id"]
        raise ValueError(f"task sample {invalid} has no pre-cutoff transactions")
    positions = history_end - 1
    counts = history_end.astype(np.float64)

    amounts = pd.to_numeric(events["amount"], errors="coerce").fillna(0.0).abs().to_numpy()
    balances = pd.to_numeric(events["balance"], errors="coerce").fillna(0.0).to_numpy()
    amount_sum = np.cumsum(amounts)
    amount_square_sum = np.cumsum(np.square(amounts))
    balance_sum = np.cumsum(balances)
    balance_square_sum = np.cumsum(np.square(balances))
    amount_total = amount_sum[positions]
    amount_mean = amount_total / counts
    amount_std = np.sqrt(np.maximum(amount_square_sum[positions] / counts - np.square(amount_mean), 0.0))
    balance_mean = balance_sum[positions] / counts
    balance_std = np.sqrt(
        np.maximum(balance_square_sum[positions] / counts - np.square(balance_mean), 0.0)
    )

    day_values = event_times.normalize().to_numpy()
    unique_days = np.unique(day_values)
    active_day_positions = np.searchsorted(unique_days, cutoff_times.normalize().to_numpy(), side="right")
    active_days = active_day_positions.astype(np.float64)
    last_times = event_times[positions]
    first_time = event_times[0]
    recent_start = event_times.searchsorted(cutoff_times - pd.Timedelta(days=30), side="right")
    recent_amount_prefix = np.concatenate(([0.0], amount_sum))
    recent_amount_total = recent_amount_prefix[history_end] - recent_amount_prefix[recent_start]

    frame = pd.DataFrame(
        {
            "sample_id": task_rows["sample_id"].astype(str).to_numpy(),
            "account_age_days": np.maximum(
                0.0, (cutoff_times - create_date).total_seconds() / 86_400.0
            ),
            "history_transaction_count": counts,
            "history_active_days": active_days,
            "history_span_days": np.maximum(
                0.0, (last_times - first_time).total_seconds() / 86_400.0
            ),
            "days_since_last_transaction": np.maximum(
                0.0, (cutoff_times - last_times).total_seconds() / 86_400.0
            ),
            "transactions_per_active_day": counts / np.maximum(active_days, 1.0),
            "amount_total_abs": amount_total,
            "amount_mean_abs": amount_mean,
            "amount_std_abs": amount_std,
            "amount_min_abs": np.minimum.accumulate(amounts)[positions],
            "amount_max_abs": np.maximum.accumulate(amounts)[positions],
            "recent_30d_transaction_count": (history_end - recent_start).astype(np.float64),
            "recent_30d_amount_abs": recent_amount_total,
            "balance_mean": balance_mean,
            "balance_std": balance_std,
            "balance_min": np.minimum.accumulate(balances)[positions],
            "balance_max": np.maximum.accumulate(balances)[positions],
            "balance_last": balances[positions],
            "balance_change": balances[positions] - balances[0],
        }
    )

    trans_type_values = np.asarray([_normalise_category_value(value) for value in events["trans_type"]])
    credit_prefix = np.concatenate(([0.0], np.cumsum(amounts * (trans_type_values == "C"))))
    credit_amount = credit_prefix[history_end]
    frame["credit_amount_abs"] = credit_amount
    frame["non_credit_amount_abs"] = np.maximum(0.0, amount_total - credit_amount)
    frame["credit_amount_share"] = credit_amount / np.maximum(amount_total, 1e-6)

    for prefix, column in (
        ("trans_type", "trans_type"),
        ("operation", "operation"),
        ("category", "category"),
    ):
        values = np.asarray([_normalise_category_value(value) for value in events[column]])
        frame[f"{prefix}_unique_count"] = _cumulative_unique_counts(values)[positions]
        for value in category_values[prefix]:
            cumulative_count = np.cumsum(values == value)
            frame[f"{prefix}_share::{value}"] = cumulative_count[positions] / counts
    return frame


def _cumulative_unique_counts(values: np.ndarray) -> np.ndarray:
    """Return the number of distinct categorical values seen at each index."""
    result = np.empty(len(values), dtype=np.float64)
    seen: set[str] = set()
    for index, value in enumerate(values):
        seen.add(str(value))
        result[index] = len(seen)
    return result


def _normalise_category_value(value: Any) -> str:
    if pd.isna(value):
        return "[MISSING]"
    text = str(value).strip()
    return text if text else "[MISSING]"
