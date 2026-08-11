"""Tests for leakage-safe downstream loan-task construction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pragma_lite.tasks.definitions import build_loan_outcome_table  # noqa: E402


class LoanOutcomeTaskTest(unittest.TestCase):
    def test_uses_strictly_pre_grant_events_and_problem_statuses(self) -> None:
        events = pd.DataFrame(
            {
                "account_id": [1, 1, 1, 2, 2, 3],
                "trans_date": [
                    "2015-01-01",
                    "2015-01-02",
                    "2015-01-03",  # grant-date event: must be excluded
                    "2015-01-01",
                    "2015-01-02",
                    "2015-01-01",
                ],
            }
        )
        loans = pd.DataFrame(
            {
                "account_id": [1, 2, 3],
                "granted_date": ["2015-01-03", "2015-01-03", "2015-01-03"],
                "loan_status": ["D", "A", "B"],
            }
        )
        manifest = pd.DataFrame(
            {"account_id": [1, 2, 3], "split": ["train", "valid", "test"]}
        )

        task = build_loan_outcome_table(
            events, loans, manifest, min_pre_grant_transactions=2
        )

        self.assertEqual(task["account_id"].tolist(), [1, 2])
        self.assertEqual(task["target"].tolist(), [1, 0])
        self.assertEqual(task["pre_grant_transaction_count"].tolist(), [2, 2])
        self.assertTrue(
            all(cutoff < datetime(2015, 1, 3) for cutoff in task["cutoff_time"])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
