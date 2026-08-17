"""Tests for cutoff-safe data supplied to the interactive explorer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import unittest

from pragma_lite.data import EventTokenizer
from pragma_lite.data.models import Transaction
from pragma_lite.demo import (
    load_demo_account_index,
    load_demo_snapshot,
    transaction_bucket_label,
    valid_demo_cutoffs,
)


def _transaction(amount: float, balance: float) -> Transaction:
    return Transaction(
        account_id=1,
        occurred_at=datetime(2015, 1, 1),
        amount=amount,
        balance_after=balance,
        transaction_type="C",
        operation="CIC",
        category="HH",
        other_bank_id="AB",
        transaction_id=1,
    )


class TokenBucketTests(unittest.TestCase):
    def test_observed_amount_bucket_is_decodable(self) -> None:
        first = _transaction(10.0, 100.0)
        second = _transaction(20.0, 200.0)
        tokenizer = EventTokenizer().fit((first, second))
        label = transaction_bucket_label(tokenizer, first, "amount_abs")
        self.assertIn("quantile", label)
        self.assertIn("[", label)


class CzechDemoIntegrationTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    PROCESSED = ROOT / "data" / "processed" / "czech_bank"

    @unittest.skipUnless((PROCESSED / "events_train.parquet").exists(), "processed Czech data absent")
    def test_future_snapshot_keeps_future_rows_out_of_model_history(self) -> None:
        index = load_demo_account_index(self.PROCESSED, "future_value")
        account_id = int(index.iloc[0]["account_id"])
        cutoff = valid_demo_cutoffs(self.PROCESSED, "future_value", account_id)[0]
        snapshot = load_demo_snapshot(self.PROCESSED, "future_value", account_id, cutoff)

        self.assertGreaterEqual(len(snapshot.history_events), 20)
        self.assertTrue((snapshot.history_events["trans_date"] <= cutoff).all())
        self.assertTrue((snapshot.future_events["trans_date"] > cutoff).all())
        self.assertTrue((snapshot.future_events["trans_date"] <= snapshot.window_end_day.isoformat()).all())
        self.assertEqual(
            len(snapshot.prepared.sample.transactions), len(snapshot.history_events)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
