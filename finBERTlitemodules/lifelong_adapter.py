"""Adapt Czech-bank account, card, and loan tables into profile milestones."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data_models import AccountState, LifelongEvent, StaticProfile
from .profile_tokenizer import ProfileTokenizer


@dataclass(frozen=True, slots=True)
class LifelongSourceTables:
    """Raw tables needed to build dated account-level profile milestones."""

    accounts: pd.DataFrame
    dispositions: pd.DataFrame
    cards: pd.DataFrame
    loans: pd.DataFrame


def read_lifelong_source_tables(source_directory: str | Path) -> LifelongSourceTables:
    """Read the four headerless TSV files required for life-long events.

    ``fin_card`` is linked to an account through ``fin_disp``.  ``fin_loan``
    already carries ``account_id``.  Transaction data is intentionally not
    loaded here; it belongs to the regular event-history branch.
    """
    source_directory = Path(source_directory)
    accounts = pd.read_csv(
        source_directory / "fin_account.tsv",
        sep="\t",
        header=None,
        names=["account_id", "district_id", "create_date", "frequency"],
        parse_dates=["create_date"],
    )
    dispositions = pd.read_csv(
        source_directory / "fin_disp.tsv",
        sep="\t",
        header=None,
        names=["disp_id", "client_id", "account_id", "disp_type"],
    )
    cards = pd.read_csv(
        source_directory / "fin_card.tsv",
        sep="\t",
        header=None,
        names=["card_id", "disp_id", "card_type", "issued_date"],
        parse_dates=["issued_date"],
    )
    loans = pd.read_csv(
        source_directory / "fin_loan.tsv",
        sep="\t",
        header=None,
        names=[
            "loan_id",
            "account_id",
            "grant_date",
            "loan_amount",
            "loan_duration_months",
            "loan_payment",
            "loan_status",
        ],
        parse_dates=["grant_date"],
    )
    return LifelongSourceTables(accounts, dispositions, cards, loans)


def build_lifelong_events(
    tables: LifelongSourceTables,
) -> dict[int, tuple[LifelongEvent, ...]]:
    """Create dated account-opening, card-issued, and loan-granted events.

    ``loan_status`` is deliberately omitted.  It is a later outcome/status
    without a timestamped history, so using it at grant time could leak future
    information into pre-training.
    """
    events_by_account: dict[int, list[LifelongEvent]] = {
        int(account_id): [] for account_id in tables.accounts["account_id"]
    }

    for row in tables.accounts.itertuples(index=False):
        account_id = int(row.account_id)
        events_by_account[account_id].append(
            LifelongEvent(
                account_id=account_id,
                occurred_at=_as_datetime(row.create_date),
                event_type="account_opened",
                fields={},
            )
        )

    card_accounts = tables.cards.merge(
        tables.dispositions[["disp_id", "account_id"]],
        on="disp_id",
        how="left",
        validate="many_to_one",
    )
    for row in card_accounts.dropna(subset=["account_id"]).itertuples(index=False):
        account_id = int(row.account_id)
        if account_id not in events_by_account:
            continue
        events_by_account[account_id].append(
            LifelongEvent(
                account_id=account_id,
                occurred_at=_as_datetime(row.issued_date),
                event_type="card_issued",
                fields={"card_type": str(row.card_type).strip()},
            )
        )

    for row in tables.loans.itertuples(index=False):
        account_id = int(row.account_id)
        if account_id not in events_by_account:
            continue
        events_by_account[account_id].append(
            LifelongEvent(
                account_id=account_id,
                occurred_at=_as_datetime(row.grant_date),
                event_type="loan_granted",
                fields={
                    "loan_amount": float(row.loan_amount),
                    "loan_duration_months": int(row.loan_duration_months),
                    "loan_payment": float(row.loan_payment),
                },
            )
        )

    return {
        account_id: tuple(
            sorted(events, key=lambda event: (event.occurred_at, event.event_type))
        )
        for account_id, events in events_by_account.items()
    }


def build_account_states(
    profile_rows: pd.DataFrame,
    tables: LifelongSourceTables,
) -> dict[int, AccountState]:
    """Combine a split's static profiles with all dated milestones per account.

    Args:
        profile_rows: One processed split, e.g. ``profile_train.parquet``.
            It determines which accounts enter the returned mapping.
        tables: Output of :func:`read_lifelong_source_tables`.

    Returns:
        ``account_id -> AccountState``.  The Dataset passes this state to
        ``build_account_sample`` at each sampled cutoff.
    """
    if not profile_rows["account_id"].is_unique:
        raise ValueError("profile_rows must contain exactly one row per account")

    created_by_account = {
        int(row.account_id): _as_datetime(row.create_date)
        for row in tables.accounts.itertuples(index=False)
    }
    lifelong_events = build_lifelong_events(tables)
    account_states: dict[int, AccountState] = {}

    for profile_row in profile_rows.to_dict(orient="records"):
        account_id = int(profile_row["account_id"])
        try:
            created_at = created_by_account[account_id]
        except KeyError as error:
            raise ValueError(f"profile account {account_id} is absent from fin_account") from error

        profile_create_date = _as_datetime(profile_row["create_date"])
        if created_at != profile_create_date:
            raise ValueError(f"account creation date mismatch for account {account_id}")

        static_profile = StaticProfile(
            account_id=account_id,
            known_from=created_at,
            fields=ProfileTokenizer.static_fields_from_profile_row(profile_row),
        )
        account_states[account_id] = AccountState(
            account_id=account_id,
            static_profile=static_profile,
            lifelong_events=lifelong_events[account_id],
        )

    return account_states


def _as_datetime(value: object) -> datetime:
    """Normalise pandas timestamps to the datetime type used by data models."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("life-long event dates must be present")
    return timestamp.to_pydatetime()
