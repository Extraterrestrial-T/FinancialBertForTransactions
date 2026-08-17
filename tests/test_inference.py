"""Tests for the public base-checkpoint plus LoRA-adapter inference contract."""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path
import tempfile
import unittest

import torch

from pragma_lite import AccountTaskPredictor
from pragma_lite.data import (
    EventTokenizer,
    LifelongEventTokenizer,
    ProfileTokenizer,
    TokenizerBundle,
    save_tokenizer_bundle,
)
from pragma_lite.data.models import LifelongEvent, Transaction
from pragma_lite.models import EventMLMDemoModel, TransformerConfig
from pragma_lite.training import AccountTaskModel, LoRAConfig, adapter_state_dict, inject_history_lora


CONFIG = TransformerConfig(d_model=32, num_heads=4, num_layers=2, ffn_dim=64, dropout=0.0)


def _profile_row() -> dict[str, object]:
    row: dict[str, object] = {
        "account_id": 101,
        "create_date": datetime(2015, 1, 1),
        "birth_date": datetime(1980, 1, 1),
        "frequency": "M",
        "gender": "M",
        "region": "Prague",
    }
    for index, name in enumerate(ProfileTokenizer._RAW_NUMERIC_FIELDS, start=1):
        row[name] = float(index)
    return row


def _transactions() -> tuple[Transaction, ...]:
    start = datetime(2020, 1, 1)
    return tuple(
        Transaction(
            account_id=101,
            occurred_at=start + timedelta(days=index),
            amount=10.0 * (index + 1),
            balance_after=100.0 + 10.0 * index,
            transaction_type="C" if index % 2 == 0 else "D",
            operation="CIC",
            category="HH",
            other_bank_id="AB",
            transaction_id=index + 1,
        )
        for index in range(3)
    )


def _lifelong_events() -> tuple[LifelongEvent, ...]:
    return (
        LifelongEvent(101, datetime(2015, 1, 1), "account_opened", {}),
        LifelongEvent(
            101,
            datetime(2019, 12, 1),
            "loan_granted",
            {"loan_amount": 500.0, "loan_duration_months": 12, "loan_payment": 45.0},
        ),
    )


def _write_artifacts(directory: Path, task: str) -> tuple[Path, Path]:
    """Create one small but fully portable base/adaptor pair for a public API test."""
    torch.manual_seed(21)
    profile = _profile_row()
    transactions = _transactions()
    lifelong_events = _lifelong_events()
    cutoff = transactions[-1].occurred_at

    event_tokenizer = EventTokenizer().fit(transaction.fields for transaction in transactions)
    profile_tokenizer = ProfileTokenizer().fit(
        [ProfileTokenizer.fields_from_profile_row(profile, cutoff)]
    )
    lifelong_tokenizer = LifelongEventTokenizer().fit_lifelong_events(lifelong_events)
    bundle = TokenizerBundle(event_tokenizer, profile_tokenizer, lifelong_tokenizer)
    checkpoint_dir = directory / "checkpoint"
    save_tokenizer_bundle(bundle, checkpoint_dir / "tokenizers")

    base_path = checkpoint_dir / "best.pt"
    base = EventMLMDemoModel(len(event_tokenizer.token_to_id), CONFIG)
    torch.save(
        {
            "model_state_dict": base.state_dict(),
            "model_config": {
                "d_model": CONFIG.d_model,
                "num_heads": CONFIG.num_heads,
                "num_layers": CONFIG.num_layers,
                "ffn_dim": CONFIG.ffn_dim,
                "dropout": CONFIG.dropout,
                "rope_base": CONFIG.rope_base,
            },
            "vocabulary_size": len(event_tokenizer.token_to_id),
            "max_events": 2,
        },
        base_path,
    )

    lora_state = inject_history_lora(base, LoRAConfig(rank=4, alpha=8.0))
    model = AccountTaskModel(base, task).eval()
    with torch.no_grad():
        # Make the two task-head checkpoints visibly distinct while leaving
        # the LoRA residual's zero-init invariant intact.
        model.head.weight.fill_(0.02 if task == "future_value" else -0.02)
        model.head.bias.fill_(0.1 if task == "future_value" else -0.1)
    adapter_path = directory / f"{task}_adapter.pt"
    torch.save(
        {
            "format_version": 1,
            "task": task,
            "base_checkpoint": str(base_path.resolve()),
            "base_checkpoint_sha256": sha256(base_path.read_bytes()).hexdigest(),
            "max_events": 2,
            "lora_config": {"rank": 4, "alpha": 8.0, "dropout": 0.0},
            "target_module_names": lora_state.target_module_names,
            "trainable_parameter_count": lora_state.trainable_parameter_count,
            "backbone_parameter_count": lora_state.backbone_parameter_count,
            "adapter_state_dict": adapter_state_dict(model),
            "task_head_state_dict": model.head.state_dict(),
        },
        adapter_path,
    )
    return base_path, adapter_path


class PublicInferenceTests(unittest.TestCase):
    def test_future_value_predictor_binds_artifacts_and_truncates_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            base_path, adapter_path = _write_artifacts(directory, "future_value")

            predictor = AccountTaskPredictor.from_checkpoints(base_path, adapter_path, device="cpu")
            prediction = predictor.predict_from_rows(
                profile_row=_profile_row(),
                lifelong_events=_lifelong_events(),
                transactions=_transactions(),
                cutoff_time=datetime(2020, 1, 3),
            )

            self.assertEqual(predictor.task, "future_value")
            self.assertEqual(prediction.account_id, 101)
            self.assertEqual(prediction.event_count, 2)
            self.assertTrue(prediction.context_truncated)
            self.assertIsNone(prediction.stress_probability)
            self.assertTrue(isfinite(prediction.predicted_log_future_volume or 0.0))
            self.assertGreaterEqual(prediction.predicted_future_volume or 0.0, 0.0)

    def test_same_base_can_bind_a_different_task_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            base_path, future_adapter = _write_artifacts(directory, "future_value")
            _, stress_adapter = _write_artifacts(directory, "cashflow_stress")

            # Both task adapters point to the same base content. Loading a
            # second predictor is the intentional adapter-swapping mechanism.
            future = AccountTaskPredictor.from_checkpoints(base_path, future_adapter, device="cpu")
            stress = AccountTaskPredictor.from_checkpoints(base_path, stress_adapter, device="cpu")
            common = {
                "profile_row": _profile_row(),
                "lifelong_events": _lifelong_events(),
                "transactions": _transactions(),
                "cutoff_time": datetime(2020, 1, 3),
            }
            self.assertIsNotNone(future.predict_from_rows(**common).predicted_future_volume)
            stress_prediction = stress.predict_from_rows(**common)
            self.assertIsNotNone(stress_prediction.stress_probability)
            self.assertGreaterEqual(stress_prediction.stress_probability or 0.0, 0.0)
            self.assertLessEqual(stress_prediction.stress_probability or 1.0, 1.0)

    def test_adapter_rejects_a_different_base_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            base_path, adapter_path = _write_artifacts(directory, "future_value")
            wrong_base = directory / "wrong.pt"
            wrong_base.write_bytes(base_path.read_bytes() + b"different-base")

            with self.assertRaisesRegex(ValueError, "different base checkpoint"):
                AccountTaskPredictor.from_checkpoints(wrong_base, adapter_path, device="cpu")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
