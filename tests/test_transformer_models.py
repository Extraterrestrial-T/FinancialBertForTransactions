"""Tests for the manual attention primitives and Event Encoder MLM slice."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pragma_lite.data.handler import (  # noqa: E402
    FinBERTLiteCzechDataset,
    apply_value_mlm_mask,
    collate_account_records,
    fit_tokenizer_bundle,
)
from pragma_lite.data.tokenizer import EventTokenizer  # noqa: E402
from pragma_lite.models.transformer import (  # noqa: E402
    EventMLMDemoModel,
    MultiHeadAttention,
    TransformerBlock,
    TransformerConfig,
)


EVENTS = ROOT / "data" / "processed" / "czech_bank" / "events_train.parquet"
PROFILES = ROOT / "data" / "processed" / "czech_bank" / "profile_train.parquet"
LIFELONG_EVENTS = ROOT / "data" / "processed" / "czech_bank" / "lifelong_events.parquet"


class AttentionPrimitiveTest(unittest.TestCase):
    def test_rope_attention_masks_padding_and_all_padded_rows(self) -> None:
        torch.manual_seed(3)
        attention = MultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
        x = torch.randn(2, 4, 16)
        mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
        positions = torch.tensor([[0.0, 2.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

        output, weights = attention(
            x, mask, rope_positions=positions, return_attention_weights=True
        )

        self.assertEqual(tuple(output.shape), (2, 4, 16))
        self.assertEqual(tuple(weights.shape), (2, 4, 4, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.equal(output[0, 2:], torch.zeros_like(output[0, 2:])))
        self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))
        self.assertTrue(torch.equal(weights[:, :, :, 2:][0], torch.zeros_like(weights[:, :, :, 2:][0])))

    def test_rope_requires_an_even_head_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "even per-head"):
            MultiHeadAttention(d_model=15, num_heads=3)

        attention_without_rope = MultiHeadAttention(
            d_model=15, num_heads=3, rope_base=None
        )
        x = torch.randn(1, 2, 15)
        with self.assertRaisesRegex(ValueError, "RoPE disabled"):
            attention_without_rope(x, rope_positions=torch.tensor([0.0, 1.0]))

    def test_block_and_event_mlm_support_a_loss_and_backward_pass(self) -> None:
        torch.manual_seed(4)
        config = TransformerConfig(d_model=32, num_heads=4, num_layers=2, ffn_dim=64, dropout=0.0)
        block = TransformerBlock(config)
        block_output = block(
            torch.randn(2, 3, 32), torch.tensor([[True, True, False], [True, False, False]])
        )
        self.assertEqual(tuple(block_output.shape), (2, 3, 32))
        self.assertTrue(torch.equal(block_output[0, 2], torch.zeros_like(block_output[0, 2])))

        vocabulary_size = len(EventTokenizer().token_to_id)
        event_key_ids = torch.tensor([4, 5, 6, 7, 8, 9]).view(1, 1, 6).expand(2, 3, -1).clone()
        event_value_ids = torch.randint(4, vocabulary_size, (2, 3, 6))
        event_mask = torch.tensor([[True, True, False], [True, False, False]])
        event_key_ids[~event_mask] = 0
        event_value_ids[~event_mask] = 0
        masked = apply_value_mlm_mask(
            {"event_value_ids": event_value_ids, "event_mask": event_mask},
            token_mask_probability=1.0,
            event_mask_probability=0.0,
            field_mask_probability=0.0,
            unk_replacement_probability=0.0,
        )
        profile_key_ids = torch.tensor([[71, 72, 73], [71, 72, 0]])
        profile_value_ids = torch.randint(4, vocabulary_size, (2, 3))
        profile_value_ids[1, 2] = 0
        profile_mask = torch.tensor([[True, True, True], [True, True, False]])
        profile_rope_time = torch.tensor([[0.0, 8.0, 16.0], [0.0, 8.0, 0.0]])
        calendar_features = torch.randn(2, 3, 4)
        calendar_features[~event_mask] = 0.0
        history_rope_time = torch.tensor([[8.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        model = EventMLMDemoModel(vocabulary_size, config)
        output = model(
            event_key_ids,
            masked["event_value_ids"],
            event_mask,
            profile_key_ids,
            profile_value_ids,
            profile_mask,
            profile_rope_time,
            history_rope_time,
            calendar_features,
        )

        self.assertEqual(tuple(output.logits.shape), (2, 3, 6, vocabulary_size))
        self.assertEqual(tuple(output.account_embedding.shape), (2, 32))
        self.assertEqual(tuple(output.event_embeddings.shape), (2, 3, 32))
        self.assertEqual(tuple(output.contextual_event_embeddings.shape), (2, 3, 32))
        self.assertTrue(torch.equal(output.event_embeddings[1, 1:], torch.zeros_like(output.event_embeddings[1, 1:])))
        self.assertTrue(torch.equal(output.field_embeddings[0, 2], torch.zeros_like(output.field_embeddings[0, 2])))
        loss = F.cross_entropy(
            output.logits.reshape(-1, vocabulary_size),
            masked["mlm_labels"].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.event_encoder.key_embedding.weight.grad)
        self.assertIsNotNone(model.profile_encoder.key_embeddings.weight.grad)
        self.assertIsNotNone(model.history_encoder.calendar_projection.weight.grad)


@unittest.skipUnless(
    EVENTS.exists() and PROFILES.exists() and LIFELONG_EVENTS.exists(),
    "processed train data or Czech milestones are missing",
)
class CzechBatchIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_events = pd.read_parquet(EVENTS)
        cls.train_profiles = pd.read_parquet(PROFILES)
        cls.tokenizers = fit_tokenizer_bundle(
            cls.train_events, cls.train_profiles, LIFELONG_EVENTS
        )

    def test_real_batch_reaches_finite_event_mlm_loss(self) -> None:
        account_ids = self.train_profiles["account_id"].head(3).tolist()
        events = self.train_events.loc[self.train_events["account_id"].isin(account_ids)]
        profiles = self.train_profiles.loc[self.train_profiles["account_id"].isin(account_ids)]
        vocabulary_size = len(EventTokenizer().token_to_id)

        with TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            events_path = temporary_directory / "events.parquet"
            profiles_path = temporary_directory / "profiles.parquet"
            events.to_parquet(events_path, index=False)
            profiles.to_parquet(profiles_path, index=False)
            dataset = FinBERTLiteCzechDataset(
                events_path,
                profiles_path,
                LIFELONG_EVENTS,
                self.tokenizers,
                max_events=16,
                random_cutoff=False,
            )
            batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_account_records)))
            masked = apply_value_mlm_mask(
                batch,
                generator=torch.Generator().manual_seed(8),
            )
            model = EventMLMDemoModel(
                vocabulary_size,
                TransformerConfig(d_model=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.0),
            )
            output = model(
                masked["event_key_ids"],
                masked["event_value_ids"],
                masked["event_mask"],
                masked["profile_key_ids"],
                masked["profile_value_ids"],
                masked["profile_mask"],
                masked["profile_rope_time"],
                masked["history_rope_time"],
                masked["calendar_features"],
            )
            loss = F.cross_entropy(
                output.logits.reshape(-1, vocabulary_size),
                masked["mlm_labels"].reshape(-1),
                ignore_index=-100,
            )
            loss.backward()

        self.assertTrue(masked["mlm_mask"].any())
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main(verbosity=2)
