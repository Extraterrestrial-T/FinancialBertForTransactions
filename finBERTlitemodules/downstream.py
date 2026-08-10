"""Leakage-safe downstream task construction for the Czech-bank experiment."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from .lifelong_adapter import read_lifelong_source_tables


PROBLEM_LOAN_STATUSES = frozenset({"B", "D"})
"""Observed problematic-loan statuses in the PKDD/Czech-bank data."""


def read_loan_outcomes(source_directory: str | Path) -> pd.DataFrame:
    """Read the target-only loan outcome fields without exposing them to encoders.

    ``loan_status`` is intentionally absent from ``LifelongEvent`` fields and
    therefore cannot enter a tokenized profile.  It is read here only after
    representation learning to construct a downstream label.
    """
    loans = read_lifelong_source_tables(source_directory).loans.copy()
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
    return task.sort_values(["split", "account_id"]).reset_index(drop=True)
