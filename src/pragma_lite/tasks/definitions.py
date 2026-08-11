"""Leakage-safe downstream task construction for the Czech-bank experiment."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

PROBLEM_LOAN_STATUSES = frozenset({"B", "D"})
"""Observed problematic-loan statuses in the PKDD/Czech-bank data."""


def read_loan_outcomes(loan_outcomes_path: str | Path) -> pd.DataFrame:
    """Read the processed target-only loan table for exploratory evaluation.

    Loan status is intentionally absent from profile milestones and event
    inputs. It becomes available only here, when a leakage-safe task table is
    constructed after representation learning.
    """
    loans = pd.read_parquet(loan_outcomes_path).copy()
    required = {"account_id", "granted_date", "loan_status"}
    if missing := required.difference(loans.columns):
        raise ValueError(f"loan outcomes are missing columns: {sorted(missing)}")
    if loans["account_id"].duplicated().any():
        raise ValueError("this task expects at most one loan outcome per account")
    loans["granted_date"] = pd.to_datetime(loans["granted_date"])
    loans["loan_status"] = loans["loan_status"].astype(str)
    return loans


def build_loan_outcome_table(
    events: pd.DataFrame,
    loans: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    min_pre_grant_transactions: int = 20,
) -> pd.DataFrame:
    """Create account-level pre-loan cutoffs and observed-trouble labels.

    The target is one for status ``B`` (finished unpaid) or ``D`` (running and
    in debt), zero for the other published statuses.  The model cutoff is one
    microsecond before the grant date: same-day transactions and the
    ``loan_granted`` profile milestone are deliberately excluded.  This makes
    the resulting record a prospective *observed repayment-trouble* probe,
    rather than a label-leaking loan-status reconstruction task.
    """
    required_event_columns = {"account_id", "trans_date"}
    required_loan_columns = {"account_id", "granted_date", "loan_status"}
    required_manifest_columns = {"account_id", "split"}
    if missing := required_event_columns.difference(events.columns):
        raise ValueError(f"events is missing required columns: {sorted(missing)}")
    if missing := required_loan_columns.difference(loans.columns):
        raise ValueError(f"loans is missing required columns: {sorted(missing)}")
    if missing := required_manifest_columns.difference(split_manifest.columns):
        raise ValueError(f"split_manifest is missing required columns: {sorted(missing)}")
    if min_pre_grant_transactions < 1:
        raise ValueError("min_pre_grant_transactions must be positive")
    if loans["account_id"].duplicated().any():
        raise ValueError("this task expects at most one loan per account")

    event_times = events[["account_id", "trans_date"]].copy()
    event_times["account_id"] = event_times["account_id"].astype(int)
    event_times["trans_date"] = pd.to_datetime(event_times["trans_date"])
    outcomes = loans[["account_id", "granted_date", "loan_status"]].copy()
    outcomes["account_id"] = outcomes["account_id"].astype(int)
    outcomes["granted_date"] = pd.to_datetime(outcomes["granted_date"])

    pre_grant_counts = (
        event_times.merge(
            outcomes[["account_id", "granted_date"]],
            on="account_id",
            how="inner",
            validate="many_to_one",
        )
        .loc[lambda frame: frame["trans_date"] < frame["granted_date"]]
        .groupby("account_id")
        .size()
        .rename("pre_grant_transaction_count")
    )
    manifest = split_manifest[["account_id", "split"]].copy()
    manifest["account_id"] = manifest["account_id"].astype(int)
    if manifest["account_id"].duplicated().any():
        raise ValueError("split_manifest must contain one row per account")

    task = outcomes.merge(manifest, on="account_id", how="inner", validate="one_to_one")
    task = task.join(pre_grant_counts, on="account_id")
    task["pre_grant_transaction_count"] = (
        task["pre_grant_transaction_count"].fillna(0).astype(int)
    )
    task = task.loc[task["pre_grant_transaction_count"] >= min_pre_grant_transactions].copy()
    task["target"] = task["loan_status"].isin(PROBLEM_LOAN_STATUSES).astype(int)
    task["cutoff_time"] = task["granted_date"].map(
        lambda value: value.to_pydatetime() - timedelta(microseconds=1)
    )
    task["sample_id"] = [
        f"loan_repayment_trouble:account:{account_id}:cutoff:{granted_date.date().isoformat()}"
        for account_id, granted_date in zip(task["account_id"], task["granted_date"], strict=True)
    ]
    return task.sort_values(["split", "account_id"]).reset_index(drop=True)


def fit_low_balance_threshold(
    train_events: pd.DataFrame,
    *,
    quantile: float = 0.10,
) -> float:
    """Fit a cash-flow-stress balance threshold on training events only.

    The returned value is part of the *task definition*, not a model feature.
    It must be fitted once from the training split and then reused unchanged
    for validation and test examples.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    if "balance" not in train_events:
        raise ValueError("train_events is missing required column: balance")
    balances = pd.to_numeric(train_events["balance"], errors="coerce").dropna()
    if balances.empty:
        raise ValueError("train_events must contain at least one numeric balance")
    return float(balances.quantile(quantile))


def build_cashflow_stress_table(
    events: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    low_balance_threshold: float,
    horizon_days: int = 60,
    min_history_transactions: int = 20,
    cutoff_stride_days: int = 90,
    observation_end: object | None = None,
) -> pd.DataFrame:
    """Create repeated account snapshots for a future low-balance task.

    Each record predicts whether the balance crosses at or below a threshold
    from just after a cutoff day through ``horizon_days`` later.  Accounts
    already at or below the threshold at cutoff are excluded: this is a
    *future deterioration* task, not a trivial detector of current stress.

    ``observation_end`` must be the common end of the source data.  Candidate
    cutoffs after ``observation_end - horizon`` are excluded so that every
    label has a complete future observation window.
    """
    if not np.isfinite(low_balance_threshold):
        raise ValueError("low_balance_threshold must be finite")
    rows = _iter_forward_window_rows(
        events,
        split_manifest,
        task_name="cashflow_stress",
        horizon_days=horizon_days,
        min_history_transactions=min_history_transactions,
        cutoff_stride_days=cutoff_stride_days,
        observation_end=observation_end,
    )
    task_rows: list[dict[str, object]] = []
    for row, history, future in rows:
        current_balance = float(history.iloc[-1]["balance"])
        if current_balance <= low_balance_threshold:
            continue
        future_balances = pd.to_numeric(future["balance"], errors="coerce").dropna()
        minimum_balance = float(future_balances.min()) if not future_balances.empty else current_balance
        task_rows.append(
            {
                **row,
                "low_balance_threshold": float(low_balance_threshold),
                "future_min_balance": minimum_balance,
                "future_transaction_count": int(len(future)),
                "target": int(minimum_balance <= low_balance_threshold),
            }
        )
    return _task_frame(task_rows)


def build_future_value_table(
    events: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    horizon_days: int = 180,
    min_history_transactions: int = 20,
    cutoff_stride_days: int = 90,
    observation_end: object | None = None,
) -> pd.DataFrame:
    """Create a future-value proxy: log transaction volume over a full window.

    This is deliberately not named customer lifetime value.  The Czech-bank
    data contains no revenue, fees, acquisition cost, or account closure
    outcome.  ``target`` is ``log1p(sum(abs(amount)))`` in the future window.
    """
    rows = _iter_forward_window_rows(
        events,
        split_manifest,
        task_name="future_value",
        horizon_days=horizon_days,
        min_history_transactions=min_history_transactions,
        cutoff_stride_days=cutoff_stride_days,
        observation_end=observation_end,
    )
    task_rows: list[dict[str, object]] = []
    for row, _, future in rows:
        future_amounts = pd.to_numeric(future["amount"], errors="coerce").dropna().abs()
        future_volume = float(future_amounts.sum())
        task_rows.append(
            {
                **row,
                "future_transaction_count": int(len(future)),
                "future_active_days": int(future["trans_date"].nunique()),
                "future_transaction_volume": future_volume,
                "target": float(np.log1p(future_volume)),
            }
        )
    return _task_frame(task_rows)


def _iter_forward_window_rows(
    events: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    task_name: str,
    horizon_days: int,
    min_history_transactions: int,
    cutoff_stride_days: int,
    observation_end: object | None,
) -> Iterator[tuple[dict[str, object], pd.DataFrame, pd.DataFrame]]:
    """Yield time-safe history/future slices for repeated-cutoff tasks."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    if min_history_transactions < 1:
        raise ValueError("min_history_transactions must be positive")
    if cutoff_stride_days < 1:
        raise ValueError("cutoff_stride_days must be positive")
    required_event_columns = {"account_id", "trans_date", "trans_id", "amount", "balance"}
    required_manifest_columns = {"account_id", "split"}
    if missing := required_event_columns.difference(events.columns):
        raise ValueError(f"events is missing required columns: {sorted(missing)}")
    if missing := required_manifest_columns.difference(split_manifest.columns):
        raise ValueError(f"split_manifest is missing required columns: {sorted(missing)}")

    prepared_events = events.copy()
    prepared_events["account_id"] = prepared_events["account_id"].astype(int)
    prepared_events["trans_date"] = pd.to_datetime(prepared_events["trans_date"])
    prepared_events = prepared_events.sort_values(["account_id", "trans_date", "trans_id"])
    manifest = split_manifest[["account_id", "split"]].copy()
    manifest["account_id"] = manifest["account_id"].astype(int)
    if manifest["account_id"].duplicated().any():
        raise ValueError("split_manifest must contain one row per account")
    split_by_account = manifest.set_index("account_id")["split"].to_dict()

    dataset_end = (
        pd.Timestamp(observation_end)
        if observation_end is not None
        else pd.Timestamp(prepared_events["trans_date"].max())
    ).normalize()
    latest_allowed_cutoff = dataset_end - pd.Timedelta(days=horizon_days)

    for account_id, group in prepared_events.groupby("account_id", sort=True):
        try:
            split = split_by_account[int(account_id)]
        except KeyError as error:
            raise ValueError(f"account {account_id} is absent from split_manifest") from error
        day_counts = group.groupby(group["trans_date"].dt.normalize(), sort=False).size()
        transaction_days = pd.DatetimeIndex(day_counts.index)
        cumulative_transaction_counts = day_counts.cumsum().to_numpy()
        transaction_times = group["trans_date"]
        first_eligible_index = int(
            np.searchsorted(cumulative_transaction_counts, min_history_transactions, side="left")
        )
        day_index = first_eligible_index
        while day_index < len(transaction_days):
            cutoff_day = pd.Timestamp(transaction_days[day_index])
            if cutoff_day > latest_allowed_cutoff:
                break
            cutoff_end = cutoff_day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            history_end = int(cumulative_transaction_counts[day_index])
            window_end = cutoff_day + pd.Timedelta(days=horizon_days)
            window_end_time = window_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            future_end = int(transaction_times.searchsorted(window_end_time, side="right"))
            history = group.iloc[:history_end]
            future = group.iloc[history_end:future_end]
            cutoff_time = cutoff_end.to_pydatetime()
            yield (
                {
                    "sample_id": f"{task_name}:account:{int(account_id)}:cutoff:{cutoff_day.date().isoformat()}",
                    "account_id": int(account_id),
                    "cutoff_time": cutoff_time,
                    "cutoff_day": cutoff_day,
                    "window_end": window_end,
                    "split": str(split),
                    "history_transaction_count": int(len(history)),
                },
                history,
                future,
            )
            next_day = cutoff_day + pd.Timedelta(days=cutoff_stride_days)
            next_index = int(transaction_days.searchsorted(next_day, side="left"))
            day_index = max(next_index, day_index + 1)


def _task_frame(task_rows: list[dict[str, object]]) -> pd.DataFrame:
    """Return a consistently ordered task table or a useful empty frame."""
    columns = [
        "sample_id",
        "account_id",
        "cutoff_time",
        "cutoff_day",
        "window_end",
        "split",
        "history_transaction_count",
        "target",
    ]
    if not task_rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(task_rows)
    if frame["sample_id"].duplicated().any():
        raise ValueError("downstream task sample_id values must be unique")
    return frame.sort_values(["split", "account_id", "cutoff_time"]).reset_index(drop=True)
