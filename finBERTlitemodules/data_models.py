"""Framework-free data contracts for the Czech-bank PRAGMA-lite pipeline.

Version one deliberately models an *account* as the user entity.  Each training
record is therefore an account history observed at a particular cutoff time.
No PyTorch tensors or token IDs belong in this module; those are produced by
later dataset and tokenizer layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from math import cos, log1p, sin, tau
from typing import Any, Iterable, Mapping, TypeAlias


FieldValue: TypeAlias = str | int | float | bool | None
FieldMap: TypeAlias = Mapping[str, FieldValue]


class Split(str, Enum):
    """Account-disjoint dataset partitions."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class Transaction:
    """One row from ``fin_trans`` after semantic column decoding."""

    account_id: int
    occurred_at: datetime
    amount: float
    balance_after: float
    transaction_type: str  # C=credit, D=debit, P=cash withdrawal
    operation: str | None  # e.g. CCW, CIC, COB, WIC, ROB
    category: str | None  # e.g. HH, IN, LO, PE, IC, IO, ST
    other_bank_id: str | None
    transaction_id: int | None = None  # sorting/join key only; never tokenized

    def __post_init__(self) -> None:
        if not self.transaction_type:
            raise ValueError("transaction_type must be present")

    @property
    def fields(self) -> dict[str, FieldValue]:
        """Structured fields that the event tokenizer may encode."""
        return {
            "transaction_type": self.transaction_type,
            "operation": self.operation,
            "category": self.category,
            "other_bank_id": self.other_bank_id,
            "amount_abs": self.amount,
            "balance_after": self.balance_after,
        }


@dataclass(frozen=True, slots=True)
class LifelongEvent:
    """An important dated state change retained in profile state.

    Examples: ``account_opened``, ``card_issued``, and ``loan_granted``.
    """

    account_id: int
    occurred_at: datetime
    event_type: str
    fields: FieldMap

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("event_type must be present")


@dataclass(frozen=True, slots=True)
class StaticProfile:
    """Attributes known from ``known_from`` onward.

    ``known_from`` makes availability explicit: a sample at an earlier cutoff
    must not receive these values.  For the initial Czech-bank pipeline, set it
    to the account creation time for account/owner/district attributes.
    """

    account_id: int
    known_from: datetime
    fields: FieldMap


@dataclass(frozen=True, slots=True)
class AccountState:
    """Source material needed to derive profile state at any cutoff."""

    account_id: int
    static_profile: StaticProfile
    lifelong_events: tuple[LifelongEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.static_profile.account_id != self.account_id:
            raise ValueError("static_profile.account_id must match account_id")
        if any(event.account_id != self.account_id for event in self.lifelong_events):
            raise ValueError("all lifelong events must belong to account_id")


@dataclass(frozen=True, slots=True)
class ProfileStateAtCutoff:
    """The profile branch input for one account at one cutoff time."""

    account_id: int
    cutoff_time: datetime
    static_fields: FieldMap
    lifelong_events: tuple[LifelongEvent, ...]


@dataclass(frozen=True, slots=True)
class AccountSample:
    """One leakage-safe model record at ``(account_id, cutoff_time)``."""

    account_id: int
    cutoff_time: datetime
    profile_state: ProfileStateAtCutoff
    transactions: tuple[Transaction, ...]
    split: Split | None = None

    def __post_init__(self) -> None:
        if self.profile_state.account_id != self.account_id:
            raise ValueError("profile_state.account_id must match account_id")
        if self.profile_state.cutoff_time != self.cutoff_time:
            raise ValueError("profile state and sample must have the same cutoff")
        if any(transaction.account_id != self.account_id for transaction in self.transactions):
            raise ValueError("all transactions must belong to account_id")
        if any(transaction.occurred_at > self.cutoff_time for transaction in self.transactions):
            raise ValueError("a sample cannot contain a transaction after its cutoff")
        if any(
            _transaction_sort_key(left) > _transaction_sort_key(right)
            for left, right in zip(self.transactions, self.transactions[1:])
        ):
            raise ValueError(
                "transactions must be ordered by (occurred_at, transaction_id)"
            )


@dataclass(frozen=True, slots=True)
class EventTemporalFeatures:
    """Continuous time inputs for the History and Event Encoder branches.

    ``history_rope_coordinates`` has one value per event: transformed seconds
    from that event to the latest event in the sample.  ``calendar_features``
    contains ``(sin_dow, cos_dow, sin_dom, cos_dom)`` for every event.

    The Czech source records dates, not times of day, so it deliberately omits
    PRAGMA's hour-of-day pair rather than inventing an intraday timestamp.
    """

    latest_event_time: datetime
    history_rope_coordinates: tuple[float, ...]
    calendar_features: tuple[tuple[float, float, float, float], ...]


def materialize_profile_state(
    account_state: AccountState,
    cutoff_time: datetime,
) -> ProfileStateAtCutoff:
    """Filter profile information to what was available at ``cutoff_time``."""
    static_fields: FieldMap
    if account_state.static_profile.known_from <= cutoff_time:
        static_fields = account_state.static_profile.fields
    else:
        static_fields = {}

    available_lifelong_events = tuple(
        sorted(
            (
                event
                for event in account_state.lifelong_events
                if event.occurred_at <= cutoff_time
            ),
            key=lambda event: event.occurred_at,
        )
    )
    return ProfileStateAtCutoff(
        account_id=account_state.account_id,
        cutoff_time=cutoff_time,
        static_fields=static_fields,
        lifelong_events=available_lifelong_events,
    )


def build_account_sample(
    account_state: AccountState,
    transactions: tuple[Transaction, ...],
    cutoff_time: datetime,
    split: Split | None = None,
) -> AccountSample:
    """Build a leakage-safe sample by retaining only history at or before cutoff."""
    history = tuple(
        sorted(
            (
                transaction
                for transaction in transactions
                if transaction.occurred_at <= cutoff_time
            ),
            key=_transaction_sort_key,
        )
    )
    return AccountSample(
        account_id=account_state.account_id,
        cutoff_time=cutoff_time,
        profile_state=materialize_profile_state(account_state, cutoff_time),
        transactions=history,
        split=split,
    )


def build_event_temporal_features(sample: AccountSample) -> EventTemporalFeatures:
    """Derive PRAGMA-style event time inputs from one ordered account sample.

    The History Encoder receives ``8 * ln(1 + seconds_to_last_event / 8)`` as
    its RoPE coordinate.  The Event Encoder receives fixed-period calendar
    features.  These values are continuous features, not vocabulary tokens.
    """
    if not sample.transactions:
        raise ValueError("cannot derive event temporal features from an empty history")

    latest_event_time = sample.transactions[-1].occurred_at
    history_rope_coordinates: list[float] = []
    calendar_features: list[tuple[float, float, float, float]] = []

    for transaction in sample.transactions:
        seconds_to_last = (latest_event_time - transaction.occurred_at).total_seconds()
        if seconds_to_last < 0:
            raise ValueError("transactions must not occur after the latest event")

        history_rope_coordinates.append(8.0 * log1p(seconds_to_last / 8.0))

        day_of_week_angle = tau * transaction.occurred_at.weekday() / 7.0
        day_of_month_angle = tau * (transaction.occurred_at.day - 1) / 31.0
        calendar_features.append(
            (
                sin(day_of_week_angle),
                cos(day_of_week_angle),
                sin(day_of_month_angle),
                cos(day_of_month_angle),
            )
        )

    return EventTemporalFeatures(
        latest_event_time=latest_event_time,
        history_rope_coordinates=tuple(history_rope_coordinates),
        calendar_features=tuple(calendar_features),
    )


def _transaction_sort_key(transaction: Transaction) -> tuple[datetime, int]:
    """Use the source ID solely as a deterministic same-day tie-breaker."""
    return (
        transaction.occurred_at,
        transaction.transaction_id if transaction.transaction_id is not None else -1,
    )


def transaction_from_row(row: Mapping[str, Any]) -> Transaction:
    """Convert one cleaned Czech-bank event row into a domain transaction.

    The processed Parquet schema uses the source names ``balance`` and
    ``trans_type``; the data model uses semantic names ``balance_after`` and
    ``transaction_type``.  Identifiers remain metadata only.
    """
    return Transaction(
        account_id=int(row["account_id"]),
        occurred_at=_as_datetime(row["trans_date"]),
        amount=float(row["amount"]),
        balance_after=float(_first_present(row, "balance_after", "balance")),
        transaction_type=str(_first_present(row, "transaction_type", "trans_type")),
        operation=_optional_text(row.get("operation")),
        category=_optional_text(row.get("category")),
        other_bank_id=_optional_text(row.get("other_bank_id")),
        transaction_id=(int(row["trans_id"]) if not _is_missing(row.get("trans_id")) else None),
    )


def iter_transactions_from_rows(
    rows: Iterable[Mapping[str, Any]] | Any,
) -> Iterable[Transaction]:
    """Yield transactions from mappings or a DataFrame-like object without copying it."""
    if hasattr(rows, "itertuples") and hasattr(rows, "columns"):
        columns = tuple(rows.columns)
        for values in rows.itertuples(index=False, name=None):
            yield transaction_from_row(dict(zip(columns, values)))
        return
    for row in rows:
        yield transaction_from_row(row)


def transaction_from_rows(rows: Iterable[Mapping[str, Any]] | Any) -> tuple[Transaction, ...]:
    """Convert DataFrame-like rows or mapping records into ordered transactions."""
    return tuple(sorted(iter_transactions_from_rows(rows), key=_transaction_sort_key))


def _as_datetime(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    raise TypeError(f"expected a date/datetime value, got {value!r}")


def _is_missing(value: Any) -> bool:
    return (
        value is None
        or type(value).__name__ == "NAType"
        or isinstance(value, float) and value != value
    )


def _optional_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    normalised = str(value).strip()
    return normalised or None


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"expected one of these source columns: {keys}")
