"""Dataset, batching, and MLM corruption for Czech-bank PRAGMA-lite records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_models import (
    AccountSample,
    build_account_sample,
    build_event_temporal_features,
    transaction_from_rows,
)
from .lifelong_adapter import build_account_states, read_lifelong_source_tables
from .lifelong_tokenizer import LifelongEventTokenizer
from .profile_tokenizer import ProfileTokenizer
from .tokenizer import EventTokenizer


@dataclass(frozen=True, slots=True)
class TokenizerBundle:
    """The three train-fitted tokenizers required for every data split."""

    event: EventTokenizer
    profile: ProfileTokenizer
    lifelong: LifelongEventTokenizer


def fit_tokenizer_bundle(
    train_events: pd.DataFrame,
    train_profiles: pd.DataFrame,
    source_directory: str | Path,
) -> TokenizerBundle:
    """Fit all tokenizer numeric boundaries from training-split data only."""
    tables = read_lifelong_source_tables(source_directory)
    account_states = build_account_states(train_profiles, tables)
    event_tokenizer = EventTokenizer().fit_numeric_fields(
        {
            "amount_abs": train_events["amount"].to_numpy(),
            "balance_after": train_events["balance"].to_numpy(),
        }
    )
    last_event_time = train_events.groupby("account_id")["trans_date"].max()
    profile_fields = [
        ProfileTokenizer.fields_from_profile_row(
            row, last_event_time.loc[int(row["account_id"])]
        )
        for row in train_profiles.to_dict(orient="records")
    ]
    profile_tokenizer = ProfileTokenizer().fit(profile_fields)
    lifelong_tokenizer = LifelongEventTokenizer().fit_lifelong_events(
        event
        for state in account_states.values()
        for event in state.lifelong_events
    )
    return TokenizerBundle(event_tokenizer, profile_tokenizer, lifelong_tokenizer)


def save_tokenizer_bundle(bundle: TokenizerBundle, directory: str | Path) -> None:
    """Persist train-fitted numeric edges for validation, test, and inference."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundle.event.save_state(directory / "event_tokenizer_state.json")
    bundle.profile.save_state(directory / "profile_tokenizer_state.json")
    bundle.lifelong.save_state(directory / "lifelong_tokenizer_state.json")


def load_tokenizer_bundle(directory: str | Path) -> TokenizerBundle:
    """Load the three tokenizer states previously saved from the train split."""
    directory = Path(directory)
    event = EventTokenizer()
    profile = ProfileTokenizer()
    lifelong = LifelongEventTokenizer()
    event.load_state(directory / "event_tokenizer_state.json")
    profile.load_state(directory / "profile_tokenizer_state.json")
    lifelong.load_state(directory / "lifelong_tokenizer_state.json")
    return TokenizerBundle(event, profile, lifelong)


class FinBERTLiteCzechDataset(Dataset[dict[str, Any]]):
    """One account-history record per index, tokenized but not MLM-masked.

    Training mode samples a reproducible random cutoff day per account and
    epoch.  Evaluation mode always uses the latest transaction day, yielding a
    stable validation/test record.  The Dataset never fits tokenizers.
    """

    def __init__(
        self,
        events_path: str | Path,
        profiles_path: str | Path,
        source_directory: str | Path,
        tokenizers: TokenizerBundle,
        *,
        max_events: int = 64,
        random_cutoff: bool = True,
        seed: int = 17,
        fixed_cutoffs_by_account: Mapping[int, datetime] | None = None,
        targets_by_account: Mapping[int, int] | None = None,
        task_table: pd.DataFrame | None = None,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        if not (
            tokenizers.event.is_fitted
            and tokenizers.profile.is_fitted
            and tokenizers.lifelong.is_fitted
        ):
            raise ValueError("all tokenizers must be fitted or loaded before Dataset creation")

        self.events = pd.read_parquet(events_path).sort_values(
            ["account_id", "trans_date", "trans_id"]
        )
        self.profiles = pd.read_parquet(profiles_path)
        self.tokenizers = tokenizers
        self.max_events = max_events
        self.random_cutoff = random_cutoff
        self.seed = seed
        self.epoch = 0
        self.fixed_cutoffs_by_account = (
            {int(account_id): cutoff for account_id, cutoff in fixed_cutoffs_by_account.items()}
            if fixed_cutoffs_by_account is not None
            else None
        )
        self.targets_by_account = (
            {int(account_id): int(target) for account_id, target in targets_by_account.items()}
            if targets_by_account is not None
            else None
        )
        if self.fixed_cutoffs_by_account is not None and random_cutoff:
            raise ValueError("fixed_cutoffs_by_account requires random_cutoff=False")
        if self.targets_by_account is not None and self.fixed_cutoffs_by_account is None:
            raise ValueError("targets_by_account requires fixed_cutoffs_by_account")
        if task_table is not None and (
            self.fixed_cutoffs_by_account is not None or self.targets_by_account is not None
        ):
            raise ValueError(
                "task_table is an alternative to fixed_cutoffs_by_account and targets_by_account"
            )
        if task_table is not None and random_cutoff:
            raise ValueError("task_table requires random_cutoff=False")

        tables = read_lifelong_source_tables(source_directory)
        self.account_states = build_account_states(self.profiles, tables)
        self.profile_by_account = {
            int(row["account_id"]): row
            for row in self.profiles.to_dict(orient="records")
        }
        self.transactions_by_account = {
            int(account_id): transaction_from_rows(group)
            for account_id, group in self.events.groupby("account_id", sort=False)
        }
        available_account_ids = set(self.account_states).intersection(self.transactions_by_account)
        self.task_rows: tuple[dict[str, Any], ...] | None = None
        if task_table is not None:
            self.task_rows = self._normalise_task_table(task_table, available_account_ids)
            self.account_ids = tuple(int(row["account_id"]) for row in self.task_rows)
        else:
            account_ids = available_account_ids
            if self.fixed_cutoffs_by_account is not None:
                account_ids.intersection_update(self.fixed_cutoffs_by_account)
            if self.targets_by_account is not None:
                account_ids.intersection_update(self.targets_by_account)
            self.account_ids = tuple(sorted(account_ids))
        if not self.account_ids:
            raise ValueError("no accounts have both profile data and transactions")

        self.cutoff_days_by_account = {
            account_id: tuple(
                sorted({transaction.occurred_at for transaction in self.transactions_by_account[account_id]})
            )
            for account_id in set(self.account_ids)
        }

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic random training windows without recreating the Dataset."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.task_rows) if self.task_rows is not None else len(self.account_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_row = self.task_rows[index] if self.task_rows is not None else None
        account_id = int(task_row["account_id"]) if task_row is not None else self.account_ids[index]
        cutoff_time = (
            task_row["cutoff_time"] if task_row is not None else self._choose_cutoff(account_id, index)
        )
        full_sample = build_account_sample(
            self.account_states[account_id],
            self.transactions_by_account[account_id],
            cutoff_time,
        )
        sample = self._truncate_sample(full_sample)
        tokenized_profile = self.tokenizers.profile.encode_profile_state(
            sample.profile_state,
            self.profile_by_account[account_id],
            self.tokenizers.lifelong,
        )
        tokenized_events = [
            self.tokenizers.event.encode(transaction)
            for transaction in sample.transactions
        ]
        temporal = build_event_temporal_features(sample)

        return {
            "account_id": account_id,
            "sample_id": (
                task_row["sample_id"]
                if task_row is not None
                else f"account:{account_id}:cutoff:{cutoff_time.isoformat()}"
            ),
            "cutoff_time": cutoff_time,
            "profile_key_ids": tokenized_profile.key_ids,
            "profile_value_ids": tokenized_profile.value_ids,
            "profile_rope_time": tokenized_profile.rope_time_coordinates,
            "event_key_ids": tuple(event.key_ids for event in tokenized_events),
            "event_value_ids": tuple(event.value_ids for event in tokenized_events),
            "history_rope_time": temporal.history_rope_coordinates,
            "calendar_features": temporal.calendar_features,
            **(
                {"target": task_row["target"]}
                if task_row is not None and "target" in task_row
                else (
                    {"target": self.targets_by_account[account_id]}
                    if self.targets_by_account is not None
                    else {}
                )
            ),
        }

    @staticmethod
    def _normalise_task_table(
        task_table: pd.DataFrame,
        available_account_ids: set[int],
    ) -> tuple[dict[str, Any], ...]:
        """Validate a labelled cutoff table, including repeated account samples.

        ``task_table`` is the general downstream-task interface.  It needs an
        ``account_id`` and ``cutoff_time`` for every row and may additionally
        carry a numeric ``target`` and a stable ``sample_id``.  Repeating an
        account at several cutoffs is intentional for future-window tasks.
        """
        required_columns = {"account_id", "cutoff_time"}
        if missing := required_columns.difference(task_table.columns):
            raise ValueError(f"task_table is missing required columns: {sorted(missing)}")
        normalised = task_table.copy()
        normalised["account_id"] = normalised["account_id"].astype(int)
        normalised["cutoff_time"] = pd.to_datetime(normalised["cutoff_time"])
        if normalised["cutoff_time"].isna().any():
            raise ValueError("task_table cutoff_time values must be present")
        if "target" in normalised and normalised["target"].isna().any():
            raise ValueError("task_table target values must be present when supplied")
        if "sample_id" not in normalised:
            normalised["sample_id"] = [
                f"account:{account_id}:cutoff:{cutoff.isoformat()}:row:{row_index}"
                for row_index, (account_id, cutoff) in enumerate(
                    zip(normalised["account_id"], normalised["cutoff_time"], strict=True)
                )
            ]
        normalised["sample_id"] = normalised["sample_id"].astype(str)
        if normalised["sample_id"].duplicated().any():
            raise ValueError("task_table sample_id values must be unique")
        missing_accounts = set(normalised["account_id"]).difference(available_account_ids)
        if missing_accounts:
            preview = sorted(missing_accounts)[:5]
            raise ValueError(f"task_table contains unavailable account IDs, e.g. {preview}")

        records = normalised.to_dict(orient="records")
        for record in records:
            cutoff = record["cutoff_time"]
            record["cutoff_time"] = (
                cutoff.to_pydatetime() if hasattr(cutoff, "to_pydatetime") else cutoff
            )
            if "target" in record and isinstance(record["target"], np.generic):
                record["target"] = record["target"].item()
        return tuple(records)

    def _choose_cutoff(self, account_id: int, index: int) -> datetime:
        if self.fixed_cutoffs_by_account is not None:
            return self.fixed_cutoffs_by_account[account_id]
        cutoff_days = self.cutoff_days_by_account[account_id]
        if not self.random_cutoff:
            return cutoff_days[-1]
        seed_sequence = np.random.SeedSequence([self.seed, self.epoch, index])
        rng = np.random.default_rng(seed_sequence)
        return cutoff_days[int(rng.integers(len(cutoff_days)))]

    def _truncate_sample(self, sample: AccountSample) -> AccountSample:
        transactions = sample.transactions[-self.max_events :]
        return AccountSample(
            account_id=sample.account_id,
            cutoff_time=sample.cutoff_time,
            profile_state=sample.profile_state,
            transactions=transactions,
            split=sample.split,
        )


# Preserve the original name while the project transitions to standard Python
# class naming.
finBERTliteCzechDataset = FinBERTLiteCzechDataset


def collate_account_records(
    records: Sequence[dict[str, Any]], *, pad_id: int = 0
) -> dict[str, Any]:
    """Pad tokenized account records into tensors for the three encoders."""
    if not records:
        raise ValueError("cannot collate an empty record list")

    batch_size = len(records)
    max_profile_tokens = max(len(record["profile_key_ids"]) for record in records)
    max_history_events = max(len(record["event_key_ids"]) for record in records)
    fields_per_event = len(records[0]["event_key_ids"][0])

    profile_key_ids = torch.full((batch_size, max_profile_tokens), pad_id, dtype=torch.long)
    profile_value_ids = torch.full_like(profile_key_ids, pad_id)
    profile_rope_time = torch.zeros((batch_size, max_profile_tokens), dtype=torch.float32)
    profile_mask = torch.zeros((batch_size, max_profile_tokens), dtype=torch.bool)

    event_key_ids = torch.full(
        (batch_size, max_history_events, fields_per_event), pad_id, dtype=torch.long
    )
    event_value_ids = torch.full_like(event_key_ids, pad_id)
    history_rope_time = torch.zeros((batch_size, max_history_events), dtype=torch.float32)
    calendar_features = torch.zeros((batch_size, max_history_events, 4), dtype=torch.float32)
    event_mask = torch.zeros((batch_size, max_history_events), dtype=torch.bool)

    for batch_index, record in enumerate(records):
        profile_length = len(record["profile_key_ids"])
        event_length = len(record["event_key_ids"])
        if any(len(event) != fields_per_event for event in record["event_key_ids"]):
            raise ValueError("all events must contain the same number of field pairs")

        profile_key_ids[batch_index, :profile_length] = torch.tensor(
            record["profile_key_ids"], dtype=torch.long
        )
        profile_value_ids[batch_index, :profile_length] = torch.tensor(
            record["profile_value_ids"], dtype=torch.long
        )
        profile_rope_time[batch_index, :profile_length] = torch.tensor(
            record["profile_rope_time"], dtype=torch.float32
        )
        profile_mask[batch_index, :profile_length] = True

        event_key_ids[batch_index, :event_length] = torch.tensor(
            record["event_key_ids"], dtype=torch.long
        )
        event_value_ids[batch_index, :event_length] = torch.tensor(
            record["event_value_ids"], dtype=torch.long
        )
        history_rope_time[batch_index, :event_length] = torch.tensor(
            record["history_rope_time"], dtype=torch.float32
        )
        calendar_features[batch_index, :event_length] = torch.tensor(
            record["calendar_features"], dtype=torch.float32
        )
        event_mask[batch_index, :event_length] = True

    batch = {
        "account_ids": torch.tensor([record["account_id"] for record in records], dtype=torch.long),
        "sample_ids": tuple(str(record["sample_id"]) for record in records),
        "cutoff_times": tuple(record["cutoff_time"] for record in records),
        "profile_key_ids": profile_key_ids,
        "profile_value_ids": profile_value_ids,
        "profile_rope_time": profile_rope_time,
        "profile_mask": profile_mask,
        "event_key_ids": event_key_ids,
        "event_value_ids": event_value_ids,
        "history_rope_time": history_rope_time,
        "calendar_features": calendar_features,
        "event_mask": event_mask,
    }
    has_targets = ["target" in record for record in records]
    if any(has_targets) and not all(has_targets):
        raise ValueError("either every record must have a target or none may have one")
    if all(has_targets):
        targets = [record["target"] for record in records]
        target_dtype = (
            torch.float32
            if any(isinstance(target, (float, np.floating)) for target in targets)
            else torch.long
        )
        batch["targets"] = torch.tensor(targets, dtype=target_dtype)
    return batch


def apply_value_mlm_mask(
    batch: dict[str, Any],
    *,
    pad_id: int = 0,
    mask_id: int = 1,
    unk_id: int = 2,
    missing_id: int = 3,
    token_mask_probability: float = 0.15,
    event_mask_probability: float = 0.10,
    field_mask_probability: float = 0.10,
    unk_replacement_probability: float = 0.05,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Corrupt event values for MLM while retaining original IDs as labels.

    Individual, full-event, and same-field-across-history selections are
    combined.  `[UNK]` replacements are input dropout and receive ``-100`` so
    they are excluded from the MLM loss, following PRAGMA's described policy.
    """
    values = batch["event_value_ids"]
    event_mask = batch["event_mask"]
    candidates = (
        event_mask.unsqueeze(-1)
        & values.ne(pad_id)
        & values.ne(missing_id)
    )
    random = lambda shape: torch.rand(shape, generator=generator, device=values.device)
    selected = (random(values.shape) < token_mask_probability) & candidates
    selected |= (
        (random(event_mask.shape).unsqueeze(-1) < event_mask_probability)
        & candidates
    )
    selected |= (
        (random((values.shape[0], 1, values.shape[2])) < field_mask_probability)
        & candidates
    )

    unk_positions = selected & (random(values.shape) < unk_replacement_probability)
    labels = torch.full_like(values, -100)
    labels[selected & ~unk_positions] = values[selected & ~unk_positions]

    masked_values = values.clone()
    masked_values[selected] = mask_id
    masked_values[unk_positions] = unk_id

    masked_batch = dict(batch)
    masked_batch["event_value_ids"] = masked_values
    masked_batch["mlm_labels"] = labels
    masked_batch["mlm_mask"] = labels.ne(-100)
    return masked_batch
