"""Adapt Czech-bank account, card, and loan tables into profile milestones."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .models import AccountState, LifelongEvent, StaticProfile
from .profile_tokenizer import ProfileTokenizer


@dataclass(frozen=True, slots=True)
class LifelongSourceTables:
    """Raw tables needed to build dated account-level profile milestones."""

    accounts: pd.DataFrame
    dispositions: pd.DataFrame
    cards: pd.DataFrame
    loans: pd.DataFrame


LIFELONG_EVENT_COLUMNS = (
    "account_id",
    "occurred_at",
    "event_type",
    "card_type",
    "loan_amount",
    "loan_duration_months",
    "loan_payment",
)
LOAN_OUTCOME_COLUMNS = (
    "loan_id",
    "account_id",
    "granted_date",
    "loan_amount",
    "loan_duration_months",
    "loan_payment",
    "loan_status",
)


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


def materialize_lifelong_events(tables: LifelongSourceTables) -> pd.DataFrame:
    """Flatten non-leaking account milestones into a portable Parquet table.

    The output intentionally excludes ``loan_status``.  It can therefore be
    committed beside the processed event/profile splits and safely consumed by
    pre-training or forward-looking tasks without the raw Teradata export.
    """
    rows: list[dict[str, object]] = []
    for account_id, events in build_lifelong_events(tables).items():
        for event in events:
            rows.append(
                {
                    "account_id": account_id,
                    "occurred_at": event.occurred_at,
                    "event_type": event.event_type,
                    "card_type": event.fields.get("card_type"),
                    "loan_amount": event.fields.get("loan_amount"),
                    "loan_duration_months": event.fields.get("loan_duration_months"),
                    "loan_payment": event.fields.get("loan_payment"),
                }
            )
    frame = pd.DataFrame(rows, columns=LIFELONG_EVENT_COLUMNS)
    return frame.sort_values(["account_id", "occurred_at", "event_type"]).reset_index(drop=True)


def materialize_loan_outcomes(tables: LifelongSourceTables) -> pd.DataFrame:
    """Return target-only loan outcomes for the optional exploratory loan task."""
    loans = tables.loans.rename(columns={"grant_date": "granted_date"}).copy()
    if loans["account_id"].duplicated().any():
        raise ValueError("this Czech-bank loan task expects at most one loan per account")
    return loans.loc[:, LOAN_OUTCOME_COLUMNS].sort_values("account_id").reset_index(drop=True)


def build_account_states_from_processed(
    profile_rows: pd.DataFrame,
    lifelong_events: pd.DataFrame,
) -> dict[int, AccountState]:
    """Materialize split-local account states without reading raw source files."""
    if not profile_rows["account_id"].is_unique:
        raise ValueError("profile_rows must contain exactly one row per account")
    required = set(LIFELONG_EVENT_COLUMNS)
    if missing := required.difference(lifelong_events.columns):
        raise ValueError(f"lifelong events are missing columns: {sorted(missing)}")

    event_rows = lifelong_events.copy()
    event_rows["account_id"] = event_rows["account_id"].astype(int)
    event_rows["occurred_at"] = pd.to_datetime(event_rows["occurred_at"])
    events_by_account: dict[int, tuple[LifelongEvent, ...]] = {}
    for account_id, group in event_rows.groupby("account_id", sort=False):
        events: list[LifelongEvent] = []
        for row in group.sort_values(["occurred_at", "event_type"]).to_dict(orient="records"):
            fields: dict[str, object] = {}
            if row["event_type"] == "card_issued" and pd.notna(row["card_type"]):
                fields["card_type"] = str(row["card_type"])
            if row["event_type"] == "loan_granted":
                for name in ("loan_amount", "loan_duration_months", "loan_payment"):
                    if pd.notna(row[name]):
                        value = row[name]
                        fields[name] = int(value) if name == "loan_duration_months" else float(value)
            events.append(
                LifelongEvent(
                    account_id=int(account_id),
                    occurred_at=_as_datetime(row["occurred_at"]),
                    event_type=str(row["event_type"]),
                    fields=fields,
                )
            )
        events_by_account[int(account_id)] = tuple(events)

    account_states: dict[int, AccountState] = {}
    for profile_row in profile_rows.to_dict(orient="records"):
        account_id = int(profile_row["account_id"])
        created_at = _as_datetime(profile_row["create_date"])
        events = events_by_account.get(account_id)
        if not events:
            raise ValueError(f"profile account {account_id} has no materialized lifelong events")
        if events[0].event_type != "account_opened" or events[0].occurred_at != created_at:
            raise ValueError(f"account creation date mismatch for account {account_id}")
        account_states[account_id] = AccountState(
            account_id=account_id,
            static_profile=StaticProfile(
                account_id=account_id,
                known_from=created_at,
                fields=ProfileTokenizer.static_fields_from_profile_row(profile_row),
            ),
            lifelong_events=events,
        )
    return account_states


def _as_datetime(value: object) -> datetime:
    """Normalise pandas timestamps to the datetime type used by data models."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("life-long event dates must be present")
    return timestamp.to_pydatetime()
