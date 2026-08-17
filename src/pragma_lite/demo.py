"""Cutoff-safe data helpers for the interactive PRAGMA-lite explorer.

The helpers deliberately distinguish three things that are easy to conflate:
the historical events supplied to the model, the future events used only to
reveal what happened after a demonstration cutoff, and the percentile buckets
used to encode individual *input* amounts and balances.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from math import log1p
from pathlib import Path
from typing import Literal

import pandas as pd

from .data.tokenizer import EventTokenizer
from .inference import PreparedAccountSample, load_processed_czech_account_sample
from .tasks import fit_low_balance_threshold


DemoTask = Literal["cashflow_stress", "future_value"]


@dataclass(frozen=True, slots=True)
class DemoTaskSpec:
    """Definition shared by a trained task adapter and its demonstration UI."""

    task: DemoTask
    label: str
    horizon_days: int
    min_history_transactions: int = 20
    cutoff_stride_days: int = 180


TASK_SPECS: dict[DemoTask, DemoTaskSpec] = {
    "cashflow_stress": DemoTaskSpec(
        task="cashflow_stress",
        label="60-day cash-flow stress",
        horizon_days=60,
    ),
    "future_value": DemoTaskSpec(
        task="future_value",
        label="180-day transaction-volume proxy",
        horizon_days=180,
    ),
}


@dataclass(frozen=True, slots=True)
class DemoSnapshot:
    """One visualized account cutoff plus future data kept outside model input."""

    task: DemoTask
    prepared: PreparedAccountSample
    split: str
    cutoff_day: date
    window_end_day: date
    history_events: pd.DataFrame
    future_events: pd.DataFrame
    low_balance_threshold: float | None
    observed_target: float | int

    @property
    def observed_future_volume(self) -> float | None:
        if self.task != "future_value":
            return None
        return float(self.future_events["amount"].abs().sum())

    @property
    def observed_future_minimum_balance(self) -> float | None:
        if self.task != "cashflow_stress":
            return None
        if self.future_events.empty:
            return float(self.history_events.iloc[-1]["balance"])
        return float(self.future_events["balance"].min())


def task_spec(task: DemoTask) -> DemoTaskSpec:
    """Return a fixed demonstration task definition."""
    try:
        return TASK_SPECS[task]
    except KeyError as error:
        raise ValueError(f"unsupported demo task {task!r}") from error


def load_demo_account_index(processed_dir: str | Path, task: DemoTask) -> pd.DataFrame:
    """Return accounts with enough history before a complete future window.

    This is an index for the UI's account picker, not a training table. Exact
    valid cutoff choices are calculated later for the selected account.
    """
    directory = Path(processed_dir)
    spec = task_spec(task)
    frames = [
        pd.read_parquet(directory / f"events_{split}.parquet", columns=["account_id", "trans_date"])
        for split in ("train", "valid", "test")
    ]
    events = pd.concat(frames, ignore_index=True)
    events["trans_date"] = pd.to_datetime(events["trans_date"]).dt.normalize()
    latest_cutoff = events["trans_date"].max() - pd.Timedelta(days=spec.horizon_days)
    eligible = events.loc[events["trans_date"].le(latest_cutoff)]
    counts = eligible.groupby("account_id").size().rename("eligible_history_events")
    manifest = pd.read_parquet(directory / "account_split_manifest.parquet", columns=["account_id", "split"])
    index = manifest.merge(counts, on="account_id", how="inner")
    return (
        index.loc[index["eligible_history_events"].ge(spec.min_history_transactions)]
        .sort_values("account_id")
        .reset_index(drop=True)
    )


def valid_demo_cutoffs(
    processed_dir: str | Path,
    task: DemoTask,
    account_id: int,
) -> tuple[datetime, ...]:
    """Return task-compatible cutoff timestamps for a single account.

    The cadence and minimum-history rule match the task-table construction.
    Cash-flow cutoffs also exclude accounts already below the training-fitted
    threshold, because that would be current-state detection rather than a
    future-stress forecast.
    """
    directory = Path(processed_dir)
    spec = task_spec(task)
    events = _read_account_events(directory, account_id)
    all_end = _observation_end(directory)
    latest_cutoff = all_end - pd.Timedelta(days=spec.horizon_days)
    days = pd.to_datetime(events["trans_date"]).dt.normalize()
    day_counts = events.groupby(days, sort=False).size()
    transaction_days = pd.DatetimeIndex(day_counts.index)
    cumulative = day_counts.cumsum().to_numpy()
    first_index = int(cumulative.searchsorted(spec.min_history_transactions, side="left"))
    if first_index >= len(transaction_days):
        return ()

    threshold = _low_balance_threshold(directory) if task == "cashflow_stress" else None
    cutoffs: list[datetime] = []
    index = first_index
    while index < len(transaction_days):
        cutoff_day = pd.Timestamp(transaction_days[index]).normalize()
        if cutoff_day > latest_cutoff:
            break
        history_end = int(cumulative[index])
        if threshold is None or float(events.iloc[history_end - 1]["balance"]) > threshold:
            cutoffs.append(_end_of_day(cutoff_day))
        next_day = cutoff_day + pd.Timedelta(days=spec.cutoff_stride_days)
        index = max(int(transaction_days.searchsorted(next_day, side="left")), index + 1)
    return tuple(cutoffs)


def load_demo_snapshot(
    processed_dir: str | Path,
    task: DemoTask,
    account_id: int,
    cutoff_time: datetime,
) -> DemoSnapshot:
    """Load one model history and hold back its observed future for the UI."""
    directory = Path(processed_dir)
    spec = task_spec(task)
    cutoff = _end_of_day(pd.Timestamp(cutoff_time))
    events = _read_account_events(directory, account_id)
    event_days = pd.to_datetime(events["trans_date"]).dt.normalize()
    cutoff_day = pd.Timestamp(cutoff).normalize()
    window_end = cutoff_day + pd.Timedelta(days=spec.horizon_days)
    history = events.loc[event_days.le(cutoff_day)].copy()
    future = events.loc[event_days.gt(cutoff_day) & event_days.le(window_end)].copy()
    if len(history) < spec.min_history_transactions:
        raise ValueError("selected cutoff has too little history for this task")

    prepared = load_processed_czech_account_sample(directory, account_id, cutoff_time=cutoff)
    split = _account_split(directory, account_id)
    if task == "future_value":
        target: float | int = float(log1p(future["amount"].abs().sum()))
        threshold = None
    else:
        threshold = _low_balance_threshold(directory)
        current_balance = float(history.iloc[-1]["balance"])
        if current_balance <= threshold:
            raise ValueError("selected cutoff is already below the stress threshold")
        future_minimum = float(future["balance"].min()) if not future.empty else current_balance
        target = int(future_minimum <= threshold)
    return DemoSnapshot(
        task=task,
        prepared=prepared,
        split=split,
        cutoff_day=cutoff_day.date(),
        window_end_day=window_end.date(),
        history_events=history,
        future_events=future,
        low_balance_threshold=threshold,
        observed_target=target,
    )


def transaction_bucket_label(tokenizer: EventTokenizer, transaction: object, field: str) -> str:
    """Describe the train-fitted token bucket for one observed input field."""
    encoded = tokenizer.encode(transaction)
    for decoded in tokenizer.decode(encoded):
        if decoded.key != field:
            continue
        if isinstance(decoded.value, dict) and decoded.value.get("kind") == "quantile_bucket":
            lower = decoded.value["lower"]
            upper = decoded.value["upper"]
            return f"quantile {decoded.value['index']:02d}: [{lower:,.2f}, {upper:,.2f}]"
        if decoded.value_token == "num:zero":
            return "exact zero token"
        return str(decoded.value)
    raise ValueError(f"transaction has no encoded field {field!r}")


def _read_account_events(directory: Path, account_id: int) -> pd.DataFrame:
    for split in ("train", "valid", "test"):
        frame = pd.read_parquet(
            directory / f"events_{split}.parquet",
            filters=[("account_id", "=", int(account_id))],
        )
        if not frame.empty:
            return frame.sort_values(["trans_date", "trans_id"]).reset_index(drop=True)
    raise ValueError(f"account {account_id} is absent from processed events")


def _observation_end(directory: Path) -> pd.Timestamp:
    latest = [
        pd.read_parquet(directory / f"events_{split}.parquet", columns=["trans_date"])["trans_date"].max()
        for split in ("train", "valid", "test")
    ]
    return pd.Timestamp(max(latest)).normalize()


def _low_balance_threshold(directory: Path) -> float:
    train_events = pd.read_parquet(directory / "events_train.parquet", columns=["balance"])
    return fit_low_balance_threshold(train_events, quantile=0.10)


def _account_split(directory: Path, account_id: int) -> str:
    manifest = pd.read_parquet(
        directory / "account_split_manifest.parquet",
        filters=[("account_id", "=", int(account_id))],
    )
    if len(manifest) != 1:
        raise ValueError(f"account {account_id} is absent from split manifest")
    return str(manifest.iloc[0]["split"])


def _end_of_day(value: pd.Timestamp | datetime) -> datetime:
    day = pd.Timestamp(value).normalize()
    return datetime.combine(day.date(), time.max)
