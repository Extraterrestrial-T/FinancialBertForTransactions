"""Integration tests for account sampling, profile tokenization, batching, and MLM."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pragma_lite.data.handler import (
    FinBERTLiteCzechDataset,
    apply_value_mlm_mask,
    collate_account_records,
    fit_tokenizer_bundle,
    load_tokenizer_bundle,
    save_tokenizer_bundle,
)


EVENTS = ROOT / "data" / "processed" / "czech_bank" / "events_train.parquet"
PROFILES = ROOT / "data" / "processed" / "czech_bank" / "profile_train.parquet"
LIFELONG_EVENTS = ROOT / "data" / "processed" / "czech_bank" / "lifelong_events.parquet"


@unittest.skipUnless(
    EVENTS.exists() and PROFILES.exists() and LIFELONG_EVENTS.exists(),
    "processed train data or lifelong milestones are missing",
)
class DataHandlerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_events = pd.read_parquet(EVENTS)
        cls.train_profiles = pd.read_parquet(PROFILES)
        cls.tokenizers = fit_tokenizer_bundle(
            cls.train_events, cls.train_profiles, LIFELONG_EVENTS
        )

    def test_dataset_batch_mask_and_saved_tokenizers(self) -> None:
        account_ids = self.train_profiles["account_id"].head(3).tolist()
        events = self.train_events.loc[
            self.train_events["account_id"].isin(account_ids)
        ]
        profiles = self.train_profiles.loc[
            self.train_profiles["account_id"].isin(account_ids)
        ]

        with TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            events_path = temporary_directory / "events.parquet"
            profiles_path = temporary_directory / "profiles.parquet"
            state_directory = temporary_directory / "tokenizer_state"
            events.to_parquet(events_path, index=False)
            profiles.to_parquet(profiles_path, index=False)

            save_tokenizer_bundle(self.tokenizers, state_directory)
            loaded = load_tokenizer_bundle(state_directory)
            dataset = FinBERTLiteCzechDataset(
                events_path,
                profiles_path,
                LIFELONG_EVENTS,
                loaded,
                max_events=64,
                random_cutoff=False,
            )

            self.assertEqual(len(dataset), len(account_ids))
            record = dataset[0]
            self.assertLessEqual(len(record["event_key_ids"]), 64)
            self.assertGreater(len(record["profile_key_ids"]), 18)

            batch = next(
                iter(
                    DataLoader(
                        dataset,
                        batch_size=2,
                        collate_fn=collate_account_records,
                    )
                )
            )
            masked = apply_value_mlm_mask(
                batch,
                generator=torch.Generator().manual_seed(4),
            )

        self.assertEqual(batch["event_key_ids"].shape, batch["event_value_ids"].shape)
        self.assertEqual(batch["profile_key_ids"].shape, batch["profile_value_ids"].shape)
        self.assertEqual(masked["mlm_labels"].shape, batch["event_value_ids"].shape)
        self.assertTrue((masked["mlm_labels"].eq(-100) | masked["mlm_labels"].ge(0)).all())
        self.assertTrue(masked["mlm_mask"].any())

    def test_task_table_supports_multiple_cutoffs_for_one_account(self) -> None:
        account_id = int(self.train_profiles.iloc[0]["account_id"])
        account_events = self.train_events.loc[
            self.train_events["account_id"].eq(account_id)
        ].sort_values(["trans_date", "trans_id"])
        self.assertGreaterEqual(len(account_events), 2)
        profiles = self.train_profiles.loc[self.train_profiles["account_id"].eq(account_id)]

        with TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            events_path = temporary_directory / "events.parquet"
            profiles_path = temporary_directory / "profiles.parquet"
            account_events.to_parquet(events_path, index=False)
            profiles.to_parquet(profiles_path, index=False)
            task_table = pd.DataFrame(
                {
                    "sample_id": ["first", "latest"],
                    "account_id": [account_id, account_id],
                    "cutoff_time": [
                        account_events.iloc[0]["trans_date"],
                        account_events.iloc[-1]["trans_date"],
                    ],
                    "target": [0, 1],
                }
            )
            dataset = FinBERTLiteCzechDataset(
                events_path,
                profiles_path,
                LIFELONG_EVENTS,
                self.tokenizers,
                max_events=64,
                random_cutoff=False,
                task_table=task_table,
            )
            batch = collate_account_records([dataset[0], dataset[1]])

        self.assertEqual(len(dataset), 2)
        self.assertEqual(batch["sample_ids"], ("first", "latest"))
        self.assertEqual(batch["targets"].tolist(), [0, 1])
        self.assertLessEqual(batch["event_mask"][0].sum(), batch["event_mask"][1].sum())


if __name__ == "__main__":
    unittest.main(verbosity=2)
