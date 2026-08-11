"""Unit tests for History Encoder-only LoRA task adaptation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch
from torch.nn import functional as F

from pragma_lite.models import EventMLMDemoModel, TransformerConfig
from pragma_lite.data import fit_tokenizer_bundle, save_tokenizer_bundle
from pragma_lite.tasks import build_cashflow_stress_table, fit_low_balance_threshold
from pragma_lite.training import (
    AccountTaskModel,
    FineTuneConfig,
    LoRAConfig,
    adapter_state_dict,
    inject_history_lora,
    load_lora_task_model,
    run_lora_finetuning,
)


VOCABULARY_SIZE = 23
CONFIG = TransformerConfig(d_model=32, num_heads=4, num_layers=2, ffn_dim=64, dropout=0.0)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "event_key_ids": torch.randint(4, VOCABULARY_SIZE, (3, 4, 6)),
        "event_value_ids": torch.randint(4, VOCABULARY_SIZE, (3, 4, 6)),
        "event_mask": torch.tensor([[True, True, True, True], [True, True, False, False], [True, False, False, False]]),
        "profile_key_ids": torch.randint(4, VOCABULARY_SIZE, (3, 5)),
        "profile_value_ids": torch.randint(4, VOCABULARY_SIZE, (3, 5)),
        "profile_mask": torch.tensor([[True, True, True, True, True], [True, True, True, False, False], [True, False, False, False, False]]),
        "profile_rope_time": torch.zeros((3, 5)),
        "history_rope_time": torch.tensor([[3.0, 2.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        "calendar_features": torch.zeros((3, 4, 4)),
        "targets": torch.tensor([0.0, 1.0, 0.0]),
    }


def _forward(backbone: EventMLMDemoModel, batch: dict[str, torch.Tensor]):
    return backbone(
        batch["event_key_ids"], batch["event_value_ids"], batch["event_mask"],
        batch["profile_key_ids"], batch["profile_value_ids"], batch["profile_mask"],
        batch["profile_rope_time"], batch["history_rope_time"], batch["calendar_features"],
    )


class HistoryLoRATests(unittest.TestCase):
    def test_zero_initialised_adapter_is_exactly_equivalent(self) -> None:
        torch.manual_seed(9)
        backbone = EventMLMDemoModel(VOCABULARY_SIZE, CONFIG).eval()
        batch = _batch()
        before = _forward(backbone, batch)

        state = inject_history_lora(backbone, LoRAConfig(rank=8, alpha=8.0))
        after = _forward(backbone.eval(), batch)

        self.assertTrue(torch.equal(before.logits, after.logits))
        self.assertTrue(torch.equal(before.account_embedding, after.account_embedding))
        self.assertGreater(state.trainable_parameter_count, 0)
        self.assertLess(state.trainable_parameter_count, state.backbone_parameter_count)
        self.assertEqual(len(state.target_module_names), CONFIG.num_layers * 5)

    def test_only_adapters_and_task_head_receive_gradients(self) -> None:
        torch.manual_seed(10)
        backbone = EventMLMDemoModel(VOCABULARY_SIZE, CONFIG)
        inject_history_lora(backbone, LoRAConfig(rank=4, alpha=8.0))
        model = AccountTaskModel(backbone, "cashflow_stress")
        batch = _batch()
        loss = F.binary_cross_entropy_with_logits(model.forward_from_batch(batch), batch["targets"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for name, parameter in model.named_parameters():
            allowed = name.startswith("head.") or name.endswith("lora_a") or name.endswith("lora_b")
            self.assertEqual(parameter.requires_grad, allowed, name)
            if not allowed:
                self.assertIsNone(parameter.grad, name)
        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in model.backbone.parameters() if parameter.requires_grad))

    def test_adapter_checkpoint_reloads_identical_predictions(self) -> None:
        torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            base_checkpoint = directory / "base.pt"
            base = EventMLMDemoModel(VOCABULARY_SIZE, CONFIG)
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
                    "vocabulary_size": VOCABULARY_SIZE,
                    "max_events": 4,
                },
                base_checkpoint,
            )
            state = inject_history_lora(base, LoRAConfig(rank=4, alpha=8.0))
            model = AccountTaskModel(base, "future_value").eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(0.01 * torch.randn_like(parameter))
            batch = _batch()
            expected = model.forward_from_batch(batch)
            adapter_checkpoint = directory / "adapter.pt"
            torch.save(
                {
                    "format_version": 1,
                    "task": "future_value",
                    "lora_config": {"rank": 4, "alpha": 8.0, "dropout": 0.0},
                    "trainable_parameter_count": state.trainable_parameter_count,
                    "adapter_state_dict": adapter_state_dict(model),
                    "task_head_state_dict": model.head.state_dict(),
                },
                adapter_checkpoint,
            )

            restored, _ = load_lora_task_model(base_checkpoint, adapter_checkpoint, device="cpu")
            actual = restored.forward_from_batch(batch)
            self.assertTrue(torch.equal(expected, actual))

    def test_regression_forward_backward_is_finite(self) -> None:
        torch.manual_seed(12)
        backbone = EventMLMDemoModel(VOCABULARY_SIZE, CONFIG)
        inject_history_lora(backbone, LoRAConfig(rank=4, alpha=8.0))
        model = AccountTaskModel(backbone, "future_value")
        batch = _batch()
        targets = torch.tensor([0.2, 1.3, 0.8])
        loss = F.smooth_l1_loss(model.forward_from_batch(batch), targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.head.bias.grad)


class CzechLoRAIntegrationTest(unittest.TestCase):
    """Exercise the committed artifacts through rank selection and final test."""

    ROOT = Path(__file__).resolve().parents[1]
    PROCESSED = ROOT / "data" / "processed" / "czech_bank"

    @unittest.skipUnless((PROCESSED / "events_train.parquet").exists(), "processed Czech data absent")
    def test_small_cashflow_run_from_processed_artifacts(self) -> None:
        processed = self.PROCESSED
        events_by_split = {
            split: pd.read_parquet(processed / f"events_{split}.parquet")
            for split in ("train", "valid", "test")
        }
        manifest = pd.read_parquet(processed / "account_split_manifest.parquet")
        all_events = pd.concat(events_by_split.values(), ignore_index=True)
        task_table = build_cashflow_stress_table(
            all_events,
            manifest,
            low_balance_threshold=fit_low_balance_threshold(events_by_split["train"]),
            horizon_days=60,
            min_history_transactions=20,
            cutoff_stride_days=365,
            observation_end=all_events["trans_date"].max(),
        )
        # Keep this integration fixture fast but retain both labels in every
        # account-disjoint split so validation AP and test metrics are defined.
        selected = []
        for split in ("train", "valid", "test"):
            split_rows = task_table.loc[task_table["split"].eq(split)]
            for label in (0, 1):
                selected.append(split_rows.loc[split_rows["target"].eq(label)].head(2))
        small_task_table = pd.concat(selected, ignore_index=True)
        self.assertEqual(set(small_task_table["split"]), {"train", "valid", "test"})
        self.assertEqual(set(small_task_table["target"]), {0, 1})

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            task_table_path = directory / "cashflow_stress_task_table.parquet"
            small_task_table.to_parquet(task_table_path, index=False)
            train_account_ids = small_task_table.loc[
                small_task_table["split"].eq("train"), "account_id"
            ].unique()
            tokenizers = fit_tokenizer_bundle(
                events_by_split["train"].loc[
                    events_by_split["train"]["account_id"].isin(train_account_ids)
                ],
                pd.read_parquet(
                    processed / "profile_train.parquet",
                    filters=[("account_id", "in", train_account_ids.tolist())],
                ),
                processed / "lifelong_events.parquet",
            )
            checkpoint_dir = directory / "checkpoint"
            save_tokenizer_bundle(tokenizers, checkpoint_dir / "tokenizers")
            checkpoint_path = checkpoint_dir / "best.pt"
            base = EventMLMDemoModel(len(tokenizers.event.token_to_id), CONFIG)
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
                    "vocabulary_size": len(tokenizers.event.token_to_id),
                    "max_events": 4,
                },
                checkpoint_path,
            )
            report = run_lora_finetuning(
                task="cashflow_stress",
                base_checkpoint_path=checkpoint_path,
                task_table_path=task_table_path,
                processed_dir=processed,
                output_dir=directory / "adapters",
                config=FineTuneConfig(
                    ranks=(4,), max_epochs=2, patience=1, batch_size=2, num_workers=0
                ),
                device="cpu",
            )
            self.assertEqual(report["adaptation"]["selected_rank"], 4)
            self.assertTrue((directory / "adapters" / "cashflow_stress_lora_report.json").exists())
            self.assertIn("average_precision", report["test_metrics"])


if __name__ == "__main__":
    unittest.main()
