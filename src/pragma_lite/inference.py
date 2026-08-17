"""Public inference contract for a PRAGMA-lite base checkpoint plus LoRA adapter.

An adapter is not a standalone model: it is a small residual over one exact
pretrained backbone. :class:`AccountTaskPredictor` binds those two artifacts,
loads their train-fitted tokenizers, verifies the checkpoint identity, and
turns one cutoff-safe account snapshot into a task prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from math import expm1
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import pandas as pd
import torch
from torch import Tensor

from .data import TokenizerBundle, collate_account_records, load_tokenizer_bundle
from .data.lifelong import build_account_states_from_processed
from .data.models import (
    AccountSample,
    AccountState,
    LifelongEvent,
    StaticProfile,
    Transaction,
    build_account_sample,
    build_event_temporal_features,
    transaction_from_rows,
)
from .data.profile_tokenizer import ProfileTokenizer
from .training.downstream import AccountTaskModel, TaskName, load_lora_task_model


@dataclass(frozen=True, slots=True)
class PreparedAccountSample:
    """A model-safe account snapshot plus the raw profile fields it needs."""

    sample: AccountSample
    profile_row: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AccountPrediction:
    """One task prediction with both model-scale and user-scale values."""

    task: TaskName
    account_id: int
    cutoff_time: datetime
    event_count: int
    context_truncated: bool
    raw_model_output: float
    stress_probability: float | None = None
    predicted_log_future_volume: float | None = None
    predicted_future_volume: float | None = None


class AccountTaskPredictor:
    """A ready-to-run base model with one verified task adapter attached.

    Construct it with :meth:`from_checkpoints`. That method verifies the base
    checkpoint SHA-256 against the adapter metadata before loading any LoRA
    tensors, so a future-value adapter cannot silently be attached to a
    different MLM backbone.
    """

    def __init__(
        self,
        model: AccountTaskModel,
        tokenizers: TokenizerBundle,
        *,
        max_events: int,
        device: torch.device,
        adapter_metadata: Mapping[str, Any],
    ) -> None:
        self.model = model.eval()
        self.tokenizers = tokenizers
        self.max_events = max_events
        self.device = device
        self.adapter_metadata = adapter_metadata
        self.task: TaskName = adapter_metadata["task"]

    @classmethod
    def from_checkpoints(
        cls,
        base_checkpoint_path: str | Path,
        adapter_checkpoint_path: str | Path,
        *,
        tokenizers_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> "AccountTaskPredictor":
        """Bind a base checkpoint, its matching adapter, and tokenizer state."""
        resolved_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        base_path = Path(base_checkpoint_path).resolve()
        model, metadata = load_lora_task_model(
            base_path, adapter_checkpoint_path, device=resolved_device
        )
        tokenizer_path = (
            Path(tokenizers_dir).resolve()
            if tokenizers_dir is not None
            else base_path.parent / "tokenizers"
        )
        tokenizers = load_tokenizer_bundle(tokenizer_path)
        return cls(
            model,
            tokenizers,
            max_events=int(metadata["max_events"]),
            device=resolved_device,
            adapter_metadata=metadata,
        )

    def predict(self, prepared: PreparedAccountSample) -> AccountPrediction:
        """Predict one account snapshot without exposing post-cutoff data."""
        original_event_count = len(prepared.sample.transactions)
        if original_event_count == 0:
            raise ValueError("an inference sample must contain at least one transaction")
        sample = AccountSample(
            account_id=prepared.sample.account_id,
            cutoff_time=prepared.sample.cutoff_time,
            profile_state=prepared.sample.profile_state,
            transactions=prepared.sample.transactions[-self.max_events :],
            split=prepared.sample.split,
        )
        record = self._tokenize_sample(sample, prepared.profile_row)
        batch = _move_tensors(collate_account_records([record]), self.device)
        with torch.no_grad():
            raw_output = float(self.model.forward_from_batch(batch).item())

        common = {
            "task": self.task,
            "account_id": sample.account_id,
            "cutoff_time": sample.cutoff_time,
            "event_count": len(sample.transactions),
            "context_truncated": original_event_count > self.max_events,
            "raw_model_output": raw_output,
        }
        if self.task == "cashflow_stress":
            probability = float(torch.sigmoid(torch.tensor(raw_output)).item())
            return AccountPrediction(**common, stress_probability=probability)

        # The regression head operates on log1p(volume). Clamp only the
        # display-scale inverse: volume itself cannot be negative.
        return AccountPrediction(
            **common,
            predicted_log_future_volume=raw_output,
            predicted_future_volume=max(0.0, expm1(raw_output)),
        )

    def predict_from_rows(
        self,
        *,
        profile_row: Mapping[str, Any],
        lifelong_events: Iterable[LifelongEvent],
        transactions: Iterable[Transaction] | Iterable[Mapping[str, Any]],
        cutoff_time: datetime | date,
    ) -> AccountPrediction:
        """Convenience path for an external application’s normalized records.

        ``profile_row`` follows the processed Czech profile schema; transaction
        mappings follow the processed event schema or callers may pass typed
        :class:`Transaction` objects. The caller supplies only events known on
        or before the cutoff—:func:`build_account_sample` enforces that again.
        """
        raw_profile = dict(profile_row)
        account_id = int(raw_profile["account_id"])
        prepared_transactions = _normalise_transactions(transactions)
        known_from = _as_datetime(raw_profile["create_date"])
        state = AccountState(
            account_id=account_id,
            static_profile=StaticProfile(
                account_id=account_id,
                known_from=known_from,
                fields=ProfileTokenizer.static_fields_from_profile_row(raw_profile),
            ),
            lifelong_events=tuple(lifelong_events),
        )
        sample = build_account_sample(
            state,
            prepared_transactions,
            _as_datetime(cutoff_time),
        )
        return self.predict(PreparedAccountSample(sample=sample, profile_row=raw_profile))

    def _tokenize_sample(
        self, sample: AccountSample, raw_profile_row: Mapping[str, Any]
    ) -> dict[str, Any]:
        profile = self.tokenizers.profile.encode_profile_state(
            sample.profile_state, dict(raw_profile_row), self.tokenizers.lifelong
        )
        events = [self.tokenizers.event.encode(transaction) for transaction in sample.transactions]
        temporal = build_event_temporal_features(sample)
        return {
            "account_id": sample.account_id,
            "sample_id": f"inference:account:{sample.account_id}:cutoff:{sample.cutoff_time.isoformat()}",
            "cutoff_time": sample.cutoff_time,
            "profile_key_ids": profile.key_ids,
            "profile_value_ids": profile.value_ids,
            "profile_rope_time": profile.rope_time_coordinates,
            "event_key_ids": tuple(event.key_ids for event in events),
            "event_value_ids": tuple(event.value_ids for event in events),
            "history_rope_time": temporal.history_rope_coordinates,
            "calendar_features": temporal.calendar_features,
        }


def load_processed_czech_account_sample(
    processed_dir: str | Path,
    account_id: int,
    *,
    cutoff_time: datetime | date | None = None,
) -> PreparedAccountSample:
    """Load one committed-Czech account snapshot for demos or local exploration."""
    processed_dir = Path(processed_dir).resolve()
    account_id = int(account_id)
    profile_frame: pd.DataFrame | None = None
    events_frame: pd.DataFrame | None = None
    for split in ("train", "valid", "test"):
        candidate = pd.read_parquet(
            processed_dir / f"profile_{split}.parquet",
            filters=[("account_id", "=", account_id)],
        )
        if candidate.empty:
            continue
        profile_frame = candidate
        events_frame = pd.read_parquet(
            processed_dir / f"events_{split}.parquet",
            filters=[("account_id", "=", account_id)],
        )
        break
    if profile_frame is None or events_frame is None or events_frame.empty:
        raise ValueError(f"account {account_id} is absent or has no processed events")

    lifelong = pd.read_parquet(
        processed_dir / "lifelong_events.parquet",
        filters=[("account_id", "=", account_id)],
    )
    states = build_account_states_from_processed(profile_frame, lifelong)
    transactions = transaction_from_rows(events_frame)
    resolved_cutoff = _as_datetime(cutoff_time) if cutoff_time is not None else transactions[-1].occurred_at
    sample = build_account_sample(states[account_id], transactions, resolved_cutoff)
    return PreparedAccountSample(
        sample=sample,
        profile_row=profile_frame.iloc[0].to_dict(),
    )


def _normalise_transactions(
    transactions: Iterable[Transaction] | Iterable[Mapping[str, Any]],
) -> tuple[Transaction, ...]:
    values = tuple(transactions)
    if not values:
        return ()
    if all(isinstance(value, Transaction) for value in values):
        return tuple(sorted(values, key=lambda value: (value.occurred_at, value.transaction_id or -1)))
    if any(isinstance(value, Transaction) for value in values):
        raise TypeError("transactions must be all Transaction objects or all row mappings")
    return transaction_from_rows(values)


def _as_datetime(value: datetime | date | Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    raise TypeError(f"expected a date or datetime, got {value!r}")


def _move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }
