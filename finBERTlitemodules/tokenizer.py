"""Structured key/value tokenizer for the Czech-bank PRAGMA-lite experiment.

This module intentionally returns integer IDs only.  The PyTorch model later
looks up and sums ``Embedding(key_ids) + Embedding(value_ids)``; IDs themselves
are never added together.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class EncodedEvent:
    """Parallel key/value IDs for one event, ready for an Event Encoder."""

    key_ids: tuple[int, ...]
    value_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.key_ids) != len(self.value_ids):
            raise ValueError("key_ids and value_ids must have the same length")


@dataclass(frozen=True, slots=True)
class DecodedField:
    """Human-readable representation of one encoded key/value pair.

    Numerical buckets cannot be decoded to the original number.  Their
    ``value`` is therefore a dictionary describing the fitted bucket interval.
    """

    key: str
    value_token: str
    value: str | float | None | dict[str, float | int | str | None]


class EventTokenizer:
    """Encode known transaction fields using the fixed training vocabulary.

    Call :meth:`fit` once with *training-split* events before encoding numeric
    fields.  Categorical tokens come from ``vocabulary.json``; numerical bin
    boundaries are learned from the supplied train events and can be persisted
    with :meth:`save_state`.
    """

    _FIELD_ALIASES = {"amount": "amount_abs"}
    _NUMERIC_FIELDS = frozenset({"amount_abs", "balance_after"})

    def __init__(self, vocabulary_path: str | Path | None = None) -> None:
        """Load the fixed key/value vocabulary.

        Args:
            vocabulary_path: Optional path to ``vocabulary.json``.  Omitting it
                loads the copy next to this module, independent of the current
                working directory.
        """
        path = Path(vocabulary_path) if vocabulary_path is not None else Path(__file__).with_name("vocabulary.json")
        with path.open("r", encoding="utf-8") as file:
            vocabulary = json.load(file)

        self.token_to_id: dict[str, int] = vocabulary["token_to_id"]
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}
        self.special_token_ids: dict[str, int] = vocabulary["special_token_ids"]

        self.field_order = tuple(
            token.removeprefix("key:")
            for token, _ in sorted(self.token_to_id.items(), key=lambda item: item[1])
            if token.startswith("key:")
        )
        self._validate_vocabulary()

        # None means ``fit``/``load_state`` has not supplied train-only edges.
        self.numeric_bin_edges: dict[str, np.ndarray | None] = {
            field: None for field in self._NUMERIC_FIELDS
        }

    @property
    def is_fitted(self) -> bool:
        """Whether all numerical fields have train-fitted bucket boundaries."""
        return all(edges is not None for edges in self.numeric_bin_edges.values())

    def fit(self, events: Iterable[Mapping[str, Any] | Any]) -> EventTokenizer:
        """Fit numerical percentile buckets from training events only.

        Args:
            events: An iterable of transaction field mappings, e.g.
                ``transaction.fields``.  A ``Transaction`` object is also
                accepted because it exposes a ``.fields`` mapping.

        Returns:
            ``self`` so setup can read ``tokenizer = EventTokenizer().fit(...)``.

        Raises:
            ValueError: If an event contains an unknown key, a numerical value
                is malformed, or a numeric field has no non-zero train values.
        """
        values_by_field: dict[str, list[float]] = {
            field: [] for field in self._NUMERIC_FIELDS
        }

        for event in events:
            for key, value in self._canonicalise_event(event).items():
                if key not in self._NUMERIC_FIELDS or self._is_missing(value):
                    continue
                numeric_value = self._numeric_value(key, value)
                if numeric_value != 0.0:
                    values_by_field[key].append(numeric_value)

        return self.fit_numeric_fields(values_by_field)

    def fit_numeric_fields(
        self, values_by_field: Mapping[str, Sequence[float | int]]
    ) -> EventTokenizer:
        """Fit numeric bins from already-extracted training columns.

        This is equivalent to :meth:`fit` for numerical fields, but avoids
        constructing a Python event object per row when fitting a large Parquet
        table.  It is useful for the regular transaction table; categorical
        vocabulary is fixed in ``vocabulary.json``.
        """
        if set(values_by_field) != self._NUMERIC_FIELDS:
            raise ValueError(
                "numeric values must be supplied for exactly "
                f"{sorted(self._NUMERIC_FIELDS)}"
            )

        quantile_probabilities = np.linspace(0.0, 1.0, self.n_numeric_bins + 1)
        for field, values in values_by_field.items():
            numeric_values = np.asarray(values, dtype=np.float64)
            if field == "amount_abs":
                numeric_values = np.abs(numeric_values)
            numeric_values = numeric_values[np.isfinite(numeric_values)]
            numeric_values = numeric_values[numeric_values != 0.0]
            if not len(numeric_values):
                raise ValueError(f"cannot fit {field}: no non-zero training values")
            self.numeric_bin_edges[field] = np.quantile(
                numeric_values, quantile_probabilities
            )
        return self

    def encode(self, event: Mapping[str, Any] | Any) -> EncodedEvent:
        """Convert one raw transaction event into aligned key/value ID tuples.

        Known categorical values absent from the training vocabulary become
        ``[UNK]``.  Explicit null values become ``[MISSING]``.  Unknown *keys*
        raise an error because they indicate a schema mismatch.
        """
        canonical_event = self._canonicalise_event(event)
        key_ids: list[int] = []
        value_ids: list[int] = []

        # Vocabulary order makes results deterministic even if a caller's dict
        # was assembled in a different order.  Absent fields are not emitted;
        # fields explicitly present with None receive [MISSING].
        for key in self.field_order:
            if key not in canonical_event:
                continue
            key_ids.append(self.token_to_id[f"key:{key}"])
            value_ids.append(self._encode_value(key, canonical_event[key]))

        return EncodedEvent(key_ids=tuple(key_ids), value_ids=tuple(value_ids))

    def decode(self, encoded_event: EncodedEvent) -> tuple[DecodedField, ...]:
        """Decode IDs for debugging; numeric buckets return interval metadata.

        This is intentionally not a lossless inverse of :meth:`encode`: once a
        number has been bucketed, its exact original value is unavailable.
        """
        decoded: list[DecodedField] = []
        for key_id, value_id in zip(encoded_event.key_ids, encoded_event.value_ids):
            key_token = self._token_for_id(key_id)
            value_token = self._token_for_id(value_id)
            if not key_token.startswith("key:"):
                raise ValueError(f"ID {key_id} is not a key token: {key_token!r}")

            key = key_token.removeprefix("key:")
            decoded.append(
                DecodedField(
                    key=key,
                    value_token=value_token,
                    value=self._decode_value(key, value_token),
                )
            )
        return tuple(decoded)

    def save_state(self, path: str | Path) -> None:
        """Save train-fitted numeric bin edges; call only after :meth:`fit`."""
        if not self.is_fitted:
            raise ValueError("fit the tokenizer before saving its numeric state")

        payload = {
            "format_version": 1,
            "n_numeric_bins": self.n_numeric_bins,
            "numeric_bin_edges": {
                field: edges.tolist()
                for field, edges in self.numeric_bin_edges.items()
                if edges is not None
            },
        }
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def load_state(self, path: str | Path) -> None:
        """Load bin edges that were fitted on the training split."""
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("n_numeric_bins") != self.n_numeric_bins:
            raise ValueError("tokenizer state has a different number of numeric bins")
        saved_edges = payload.get("numeric_bin_edges", {})
        if set(saved_edges) != self._NUMERIC_FIELDS:
            raise ValueError("tokenizer state must contain edges for every numeric field")

        self.numeric_bin_edges = {
            field: self._validate_edges(field, np.asarray(edges, dtype=np.float64))
            for field, edges in saved_edges.items()
        }

    def _encode_value(self, key: str, value: Any) -> int:
        if self._is_missing(value):
            return self.special_token_ids["[MISSING]"]

        if key in self._NUMERIC_FIELDS:
            numeric_value = self._numeric_value(key, value)
            if numeric_value == 0.0:
                return self.token_to_id["num:zero"]

            edges = self.numeric_bin_edges[key]
            if edges is None:
                raise ValueError(f"fit or load numeric bins before encoding {key}")
            # Internal quantile boundaries yield an index in 0..n_bins-1.
            bucket = int(np.searchsorted(edges[1:-1], numeric_value, side="right"))
            bucket = min(max(bucket, 0), self.n_numeric_bins - 1)
            return self.token_to_id[f"num:quantile:{bucket:02d}"]

        category_token = f"cat:{key}:{str(value).strip()}"
        return self.token_to_id.get(category_token, self.special_token_ids["[UNK]"])

    def _decode_value(self, key: str, value_token: str) -> str | float | None | dict[str, float | int | str | None]:
        if value_token == "[MISSING]":
            return None
        if value_token == "[UNK]":
            return "[UNK]"
        if value_token == "num:zero":
            return 0.0
        if value_token.startswith("num:quantile:"):
            bucket = int(value_token.rsplit(":", maxsplit=1)[1])
            edges = self.numeric_bin_edges.get(key)
            lower = upper = None
            if edges is not None:
                lower = float(edges[bucket])
                upper = float(edges[bucket + 1])
            return {
                "kind": "quantile_bucket",
                "index": bucket,
                "lower": lower,
                "upper": upper,
            }
        if value_token.startswith(f"cat:{key}:"):
            return value_token.removeprefix(f"cat:{key}:")
        return value_token

    def _canonicalise_event(self, event: Mapping[str, Any] | Any) -> dict[str, Any]:
        fields = event if isinstance(event, Mapping) else getattr(event, "fields", None)
        if not isinstance(fields, Mapping):
            raise TypeError("event must be a field mapping or expose a .fields mapping")

        canonical: dict[str, Any] = {}
        for raw_key, value in fields.items():
            key = self._FIELD_ALIASES.get(raw_key, raw_key)
            if f"key:{key}" not in self.token_to_id:
                raise ValueError(f"unknown key {raw_key!r}; vocabulary/schema must be updated")
            if key in canonical:
                raise ValueError(f"event provides duplicate aliases for key {key!r}")
            canonical[key] = value
        return canonical

    def _numeric_value(self, key: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{key} must be a finite number, got {value!r}")
        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise ValueError(f"{key} must be finite, got {value!r}")
        return abs(numeric_value) if key == "amount_abs" else numeric_value

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None or isinstance(value, str) and not value.strip():
            return True
        # ``pandas.NA`` may appear when a Parquet row is converted to a dict.
        # Keep this module pandas-free while recognising that scalar sentinel.
        if type(value).__name__ == "NAType":
            return True
        return isinstance(value, (float, np.floating)) and np.isnan(value)

    def _token_for_id(self, token_id: int) -> str:
        try:
            return self.id_to_token[token_id]
        except KeyError as error:
            raise ValueError(f"unknown token ID {token_id}") from error

    def _validate_vocabulary(self) -> None:
        required_specials = {"[PAD]", "[MASK]", "[UNK]", "[MISSING]"}
        if not required_specials <= self.special_token_ids.keys():
            raise ValueError("vocabulary is missing one or more required special tokens")
        if not self._NUMERIC_FIELDS <= set(self.field_order):
            raise ValueError("vocabulary is missing a required numeric key token")
        if "num:zero" not in self.token_to_id:
            raise ValueError("vocabulary is missing num:zero")

        expected = [f"num:quantile:{index:02d}" for index in range(self.n_numeric_bins)]
        if any(token not in self.token_to_id for token in expected):
            raise ValueError("numeric quantile tokens must be contiguous and zero-padded")

    @property
    def n_numeric_bins(self) -> int:
        """Number of shared numerical percentile-bucket tokens in the vocab."""
        return sum(token.startswith("num:quantile:") for token in self.token_to_id)

    def _validate_edges(self, field: str, edges: np.ndarray) -> np.ndarray:
        if edges.shape != (self.n_numeric_bins + 1,):
            raise ValueError(f"{field} has an invalid number of bin edges")
        if not np.isfinite(edges).all() or np.any(np.diff(edges) < 0):
            raise ValueError(f"{field} bin edges must be finite and non-decreasing")
        return edges
