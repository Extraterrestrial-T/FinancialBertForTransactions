"""Integration check for dated account/card/loan profile milestones."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pragma_lite.data.models import materialize_profile_state
from pragma_lite.data.lifelong import (
    build_account_states_from_processed,
)
from pragma_lite.data.lifelong_tokenizer import LifelongEventTokenizer
from pragma_lite.data.profile_tokenizer import ProfileTokenizer


TRAIN_PROFILES = ROOT / "data" / "processed" / "czech_bank" / "profile_train.parquet"
LIFELONG_EVENTS = ROOT / "data" / "processed" / "czech_bank" / "lifelong_events.parquet"
CUTOFF = pd.Timestamp("2018-12-31")


@unittest.skipUnless(
    LIFELONG_EVENTS.exists() and TRAIN_PROFILES.exists(),
    "processed Czech milestones or training profiles are missing",
)
class LifelongAdapterTest(unittest.TestCase):
    def test_build_and_tokenize_profile_milestones(self) -> None:
        all_profiles = pd.read_parquet(TRAIN_PROFILES)
        all_lifelong_events = pd.read_parquet(LIFELONG_EVENTS)
        eligible_account_ids = (
            all_lifelong_events.groupby("account_id").size().loc[lambda counts: counts.gt(1)]
            .head(12)
            .index
        )
        profiles = all_profiles.loc[all_profiles["account_id"].isin(eligible_account_ids)]
        lifelong_events = all_lifelong_events.loc[
            all_lifelong_events["account_id"].isin(eligible_account_ids)
        ]
        account_states = build_account_states_from_processed(profiles, lifelong_events)

        all_events = [
            event
            for account_state in account_states.values()
            for event in account_state.lifelong_events
        ]
        self.assertEqual(len(account_states), len(profiles))
        self.assertTrue(any(event.event_type == "account_opened" for event in all_events))
        self.assertTrue(any(event.event_type == "card_issued" for event in all_events))
        self.assertTrue(any(event.event_type == "loan_granted" for event in all_events))

        profile_tokenizer = ProfileTokenizer().fit(
            [
                ProfileTokenizer.fields_from_profile_row(row, CUTOFF)
                for row in profiles.to_dict(orient="records")
            ]
        )
        lifelong_tokenizer = LifelongEventTokenizer().fit_lifelong_events(all_events)

        account_id = next(
            account_id
            for account_id, state in account_states.items()
            if len(state.lifelong_events) > 1
        )
        raw_profile = profiles.loc[profiles["account_id"].eq(account_id)].iloc[0].to_dict()
        profile_state = materialize_profile_state(
            account_states[account_id], CUTOFF.to_pydatetime()
        )
        tokenized = profile_tokenizer.encode_profile_state(
            profile_state, raw_profile, lifelong_tokenizer
        )

        self.assertEqual(len(tokenized.key_ids), len(tokenized.value_ids))
        self.assertEqual(len(tokenized.key_ids), len(tokenized.rope_time_coordinates))
        self.assertGreater(len(tokenized.key_ids), 18)
        self.assertEqual(tokenized.rope_time_coordinates[:18], (0.0,) * 18)


if __name__ == "__main__":
    unittest.main(verbosity=2)
