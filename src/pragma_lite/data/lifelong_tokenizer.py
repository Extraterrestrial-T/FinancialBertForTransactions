"""Tokenizer for dated profile milestones such as cards and loans."""

from __future__ import annotations

from collections.abc import Iterable

from .models import LifelongEvent
from .tokenizer import EncodedEvent, EventTokenizer


class LifelongEventTokenizer(EventTokenizer):
    """Encode profile events that are timestamped but not transactions.

    Every event receives a ``profile_event_type`` field in addition to its
    event-specific fields.  It shares the vocabulary and percentile-value
    tokens with the other tokenizers, while fitting its own loan-field
    boundaries from training-split lifetime events only.
    """

    _FIELD_ALIASES: dict[str, str] = {}
    _NUMERIC_FIELDS = frozenset(
        {"loan_amount", "loan_duration_months", "loan_payment"}
    )

    @staticmethod
    def fields_from_lifelong_event(event: LifelongEvent) -> dict[str, object]:
        """Add the required event-type field to an event's semantic fields."""
        return {"profile_event_type": event.event_type, **event.fields}

    def fit_lifelong_events(
        self, events: Iterable[LifelongEvent]
    ) -> "LifelongEventTokenizer":
        """Fit loan numeric buckets using only training-split lifetime events."""
        self.fit(self.fields_from_lifelong_event(event) for event in events)
        return self

    def encode_lifelong_event(self, event: LifelongEvent) -> EncodedEvent:
        """Encode one dated milestone; its date is handled by the profile branch."""
        return self.encode(self.fields_from_lifelong_event(event))
