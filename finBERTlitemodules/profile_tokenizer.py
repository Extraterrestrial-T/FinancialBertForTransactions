"""Tokenizer for the static user/profile-state branch of PRAGMA-lite."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from math import log1p
from typing import Any

from .data_models import ProfileStateAtCutoff
from .lifelong_tokenizer import LifelongEventTokenizer
from .tokenizer import EventTokenizer


@dataclass(frozen=True, slots=True)
class TokenizedProfileState:
    """Flattened profile tokens and their RoPE time coordinates.

    Static fields have a time coordinate of zero.  Each pair belonging to a
    dated life-long event receives the same log-seconds-since-event coordinate.
    The model adds its own learned [USR] vector; it is not included here.
    """

    key_ids: tuple[int, ...]
    value_ids: tuple[int, ...]
    rope_time_coordinates: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.key_ids)
            == len(self.value_ids)
            == len(self.rope_time_coordinates)
        ):
            raise ValueError("profile key, value, and time arrays must align")


class ProfileTokenizer(EventTokenizer):
    """Encode a leakage-safe static profile at a particular cutoff time.

    This initial version covers account/owner attributes and district
    demographics.  Account and client IDs, district IDs, raw birth dates, and
    raw creation dates are deliberately excluded.  The latter two become the
    derived ``account_age_days`` and ``owner_age_years`` features instead.

    Dated life-long profile events are encoded by ``LifelongEventTokenizer``
    and flattened alongside these static pairs by :meth:`encode_profile_state`.
    """

    _FIELD_ALIASES = {
        "frequency": "account_frequency",
        "gender": "owner_gender",
    }
    _NUMERIC_FIELDS = frozenset(
        {
            "account_age_days",
            "owner_age_years",
            "num_inhabitants",
            "num_municipalities_gt499",
            "num_municipalities_500to1999",
            "num_municipalities_2000to9999",
            "num_municipalities_gt10000",
            "num_cities",
            "ratio_urban",
            "average_salary",
            "unemployment_rate95",
            "unemployment_rate96",
            "num_entrep_per1000",
            "num_crimes95",
            "num_crimes96",
        }
    )
    _RAW_NUMERIC_FIELDS = (
        "num_inhabitants",
        "num_municipalities_gt499",
        "num_municipalities_500to1999",
        "num_municipalities_2000to9999",
        "num_municipalities_gt10000",
        "num_cities",
        "ratio_urban",
        "average_salary",
        "unemployment_rate95",
        "unemployment_rate96",
        "num_entrep_per1000",
        "num_crimes95",
        "num_crimes96",
    )

    @classmethod
    def static_fields_from_profile_row(
        cls, profile_row: dict[str, Any]
    ) -> dict[str, Any]:
        """Return stable profile fields without dates or cutoff-derived ages."""
        fields: dict[str, Any] = {
            "account_frequency": profile_row.get("frequency"),
            "owner_gender": profile_row.get("gender"),
            "region": profile_row.get("region"),
        }
        fields.update(
            {field: profile_row.get(field) for field in cls._RAW_NUMERIC_FIELDS}
        )
        return fields

    @classmethod
    def fields_from_profile_row(
        cls,
        profile_row: dict[str, Any],
        cutoff_time: datetime | date,
    ) -> dict[str, Any]:
        """Create the model-safe static fields for one profile row and cutoff.

        ``cutoff_time`` must be on or after the account creation date.  The
        profile values returned here are what should be passed to ``fit`` or
        ``encode``—not the unfiltered raw Parquet row.
        """
        cutoff = cls._as_datetime(cutoff_time)
        created = cls._as_datetime(profile_row["create_date"])
        if created > cutoff:
            raise ValueError("profile cutoff precedes the account creation date")

        birth_value = profile_row.get("birth_date")
        if cls._is_missing(birth_value):
            owner_age_years: float | None = None
        else:
            birth_date = cls._as_datetime(birth_value)
            if birth_date > cutoff:
                raise ValueError("profile cutoff precedes the owner's birth date")
            owner_age_years = (cutoff - birth_date).days / 365.2425

        fields = cls.static_fields_from_profile_row(profile_row)
        fields.update(
            {
            "account_age_days": float((cutoff - created).days),
            "owner_age_years": owner_age_years,
            }
        )
        return fields

    def encode_profile_state(
        self,
        profile_state: ProfileStateAtCutoff,
        raw_profile_row: dict[str, Any],
        lifelong_tokenizer: LifelongEventTokenizer,
    ) -> TokenizedProfileState:
        """Combine static profile pairs and dated life-long event pairs.

        ``profile_state`` must come from ``materialize_profile_state`` or
        ``build_account_sample``.  Its life-long events have therefore already
        been filtered to the sample cutoff, preventing future-event leakage.
        """
        raw_account_id = raw_profile_row.get("account_id")
        if raw_account_id is not None and int(raw_account_id) != profile_state.account_id:
            raise ValueError("raw profile row does not match profile_state account")

        static_encoded = self.encode(
            self.fields_from_profile_row(raw_profile_row, profile_state.cutoff_time)
        )
        key_ids = list(static_encoded.key_ids)
        value_ids = list(static_encoded.value_ids)
        rope_times = [0.0] * len(static_encoded.key_ids)

        for event in profile_state.lifelong_events:
            elapsed_seconds = (profile_state.cutoff_time - event.occurred_at).total_seconds()
            if elapsed_seconds < 0:
                raise ValueError("profile state contains a future life-long event")
            event_time = 8.0 * log1p(elapsed_seconds / 8.0)
            encoded_event = lifelong_tokenizer.encode_lifelong_event(event)
            key_ids.extend(encoded_event.key_ids)
            value_ids.extend(encoded_event.value_ids)
            rope_times.extend([event_time] * len(encoded_event.key_ids))

        return TokenizedProfileState(
            key_ids=tuple(key_ids),
            value_ids=tuple(value_ids),
            rope_time_coordinates=tuple(rope_times),
        )

    @staticmethod
    def _as_datetime(value: datetime | date | Any) -> datetime:
        """Accept Python dates plus pandas Timestamp values without pandas import."""
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, time.min)
        raise TypeError(f"expected a date/datetime value, got {value!r}")
