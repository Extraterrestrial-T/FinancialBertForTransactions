"""Small end-to-end check for static user/profile-state tokenization.

Run from the repository root:
    .\\FinBert\\Scripts\\python.exe -m unittest tests.test_profile_tokenizer_smoke -v
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finBERTlitemodules.profile_tokenizer import ProfileTokenizer


TRAIN_PROFILES = ROOT / "data" / "processed" / "czech_bank" / "profile_train.parquet"
RAW_PROFILE_COLUMNS = [
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
CUTOFF = pd.Timestamp("2018-12-31")


@unittest.skipUnless(TRAIN_PROFILES.exists(), f"missing training file: {TRAIN_PROFILES}")
class ProfileTokenizerSmokeTest(unittest.TestCase):
    def test_fit_encode_decode_and_reload(self) -> None:
        frame = pd.read_parquet(TRAIN_PROFILES, columns=RAW_PROFILE_COLUMNS).head(2_048)
        profile_fields = [
            ProfileTokenizer.fields_from_profile_row(row, CUTOFF)
            for row in frame.to_dict(orient="records")
        ]

        tokenizer = ProfileTokenizer().fit(profile_fields)
        encoded = tokenizer.encode(profile_fields[0])
        decoded = tokenizer.decode(encoded)

        self.assertTrue(tokenizer.is_fitted)
        self.assertEqual(len(encoded.key_ids), 18)
        self.assertEqual(decoded[0].key, "account_frequency")
        self.assertEqual(decoded[-1].key, "num_crimes96")
        self.assertEqual(decoded[3].key, "account_age_days")
        self.assertEqual(decoded[3].value["kind"], "quantile_bucket")

        with TemporaryDirectory() as temporary_directory:
            state_path = Path(temporary_directory) / "profile_tokenizer_state.json"
            tokenizer.save_state(state_path)
            restored = ProfileTokenizer()
            restored.load_state(state_path)
            self.assertEqual(restored.encode(profile_fields[0]), encoded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
