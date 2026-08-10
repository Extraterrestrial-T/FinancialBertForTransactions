"""Small end-to-end check for the PRAGMA-lite transaction tokenizer.

Run from the repository root:
    .\\FinBert\\Scripts\\python.exe -m unittest tests.test_tokenizer_smoke -v
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
# Permit both ``python -m unittest ...`` from the repository root and direct
# execution of this file from an IDE / notebook terminal.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finBERTlitemodules.tokenizer import EventTokenizer
from finBERTlitemodules.profile_tokenizer import ProfileTokenizer


TRAIN_EVENTS = ROOT / "data" / "processed" / "czech_bank" / "events_train.parquet"
TRAIN_PROFILES = ROOT / "data" / "processed" / "czech_bank" / "profile_train.parquet"
EVENT_COLUMNS = [
    "account_id",
    "trans_date",
    "trans_id",
    "trans_type",
    "operation",
    "category",
    "other_bank_id",
    "amount",
    "balance",
]
PROFILE_COLUMNS = [
    "account_id",
    "create_date",
    "birth_date",
    "frequency",
    "gender",
    "region",
    "num_inhabitants",
    "num_municipalities_gt499",
    "num_municipalities_500to1999",
    "num_municipalities_2000to9999",
    "num_municipalities_gt10000",
    "num_cities",
    "ratio_urban",
    "average_salary",
    "unemployment_rate95",
    "unemployment_rate96",
    "num_entrep_per1000",
    "num_crimes95",
    "num_crimes96",
]


def transaction_fields(row: dict[str, object]) -> dict[str, object]:
    """Match the field names exposed by ``Transaction.fields``."""
    return {
        "transaction_type": row["trans_type"],
        "operation": row["operation"],
        "category": row["category"],
        "other_bank_id": row["other_bank_id"],
        "amount": row["amount"],
        "balance_after": row["balance"],
    }


@unittest.skipUnless(TRAIN_EVENTS.exists(), f"missing training file: {TRAIN_EVENTS}")
class TokenizerSmokeTest(unittest.TestCase):
    def test_fit_encode_decode_and_reload(self) -> None:
        # This is deliberately a small sample: enough to exercise real values,
        # but fast enough to run while developing the tokenizer.
        frame = pd.read_parquet(TRAIN_EVENTS, columns=EVENT_COLUMNS).head(2_048)
        events = [transaction_fields(row) for row in frame.to_dict(orient="records")]

        tokenizer = EventTokenizer().fit(events)
        encoded = tokenizer.encode(events[0])
        decoded = tokenizer.decode(encoded)

        print("\nExample raw transaction")
        for key, value in events[0].items():
            print(f"  {key}: {value!r}")
        print("\nToken pairs: key token + value token")
        for decoded_field, key_id, value_id in zip(
            decoded, encoded.key_ids, encoded.value_ids
        ):
            print(
                f"  {decoded_field.key}: "
                f"{key_id} ({tokenizer.id_to_token[key_id]}) + "
                f"{value_id} ({tokenizer.id_to_token[value_id]})"
            )
        print("\nDecoded view")
        for field in decoded:
            print(f"  {field.key}: {field.value!r}")

        self.assertTrue(tokenizer.is_fitted)
        self.assertEqual(len(encoded.key_ids), 6)
        self.assertEqual(len(encoded.key_ids), len(encoded.value_ids))
        self.assertEqual(decoded[0].key, "transaction_type")
        self.assertEqual(decoded[-1].key, "balance_after")

        with TemporaryDirectory() as temporary_directory:
            state_path = Path(temporary_directory) / "tokenizer_state.json"
            tokenizer.save_state(state_path)

            restored = EventTokenizer()
            restored.load_state(state_path)
            self.assertEqual(restored.encode(events[0]), encoded)

    @unittest.skipUnless(TRAIN_PROFILES.exists(), f"missing training file: {TRAIN_PROFILES}")
    def test_show_ten_tokenized_user_transaction_pairs(self) -> None:
        """Print ten profile + transaction examples for visual inspection."""
        profile_frame = pd.read_parquet(TRAIN_PROFILES, columns=PROFILE_COLUMNS).head(2_048)
        event_frame = pd.read_parquet(TRAIN_EVENTS, columns=EVENT_COLUMNS)
        latest_events = (
            event_frame.sort_values(["account_id", "trans_date", "trans_id"])
            .groupby("account_id", as_index=False)
            .tail(1)
            .set_index("account_id")
        )

        # Fit both tokenizers only on train-split rows.  Each profile uses its
        # own latest event as cutoff, just as a record-level user embedding
        # would at an evaluation point.
        profile_fit_fields = [
            ProfileTokenizer.fields_from_profile_row(
                row, latest_events.loc[row["account_id"], "trans_date"]
            )
            for row in profile_frame.to_dict(orient="records")
        ]
        profile_tokenizer = ProfileTokenizer().fit(profile_fit_fields)
        event_tokenizer = EventTokenizer().fit(
            [transaction_fields(row) for row in event_frame.head(2_048).to_dict(orient="records")]
        )

        selected_accounts = profile_frame["account_id"].head(10).tolist()
        examples = latest_events.loc[selected_accounts]

        self.assertEqual(len(examples), 10)
        print("\nTen tokenized user-profile + latest-transaction pairs")
        for profile_row in profile_frame.head(10).to_dict(orient="records"):
            account_id = profile_row["account_id"]  # join/display only; never tokenized
            transaction_row = examples.loc[account_id].to_dict()
            profile_fields = ProfileTokenizer.fields_from_profile_row(
                profile_row, transaction_row["trans_date"]
            )

            profile_encoded = profile_tokenizer.encode(profile_fields)
            transaction_encoded = event_tokenizer.encode(transaction_fields(transaction_row))
            print(f"\nAccount {account_id}")
            print(f"  profile key/value IDs: {list(zip(profile_encoded.key_ids, profile_encoded.value_ids))}")
            print(f"  event   key/value IDs: {list(zip(transaction_encoded.key_ids, transaction_encoded.value_ids))}")

            self.assertEqual(len(profile_encoded.key_ids), 18)
            self.assertEqual(len(transaction_encoded.key_ids), 6)


if __name__ == "__main__":
    unittest.main(verbosity=3)
