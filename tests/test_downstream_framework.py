"""Unit tests for repeated-cutoff tasks and train-only tabular features."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from importlib.util import find_spec

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finBERTlitemodules.downstream import (  # noqa: E402
    build_cashflow_stress_table,
    build_future_value_table,
    fit_low_balance_threshold,
)
from finBERTlitemodules.downstream_features import (  # noqa: E402
    build_account_history_features,
    fit_tabular_feature_schema,
)
from finBERTlitemodules.downstream_evaluation import (  # noqa: E402
    binary_classification_metrics,
    fit_binary_logistic_benchmark,
    fit_platt_probability_calibrator,
    fit_ridge_regression_benchmark,
)


class RepeatedCutoffTaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = pd.DataFrame(
            {
                "trans_id": [1, 2, 3, 4, 5, 6, 7, 8],
                "account_id": [1, 1, 1, 1, 1, 1, 2, 2],
                "trans_date": pd.to_datetime(
                    [
                        "2018-01-01",
                        "2018-01-02",
                        "2018-01-03",
                        "2018-01-05",
                        "2018-01-08",
                        "2018-01-10",
                        "2018-01-01",
                        "2018-01-03",
                    ]
                ),
                "amount": [10.0, 10.0, 10.0, 20.0, 30.0, 40.0, 5.0, 5.0],
                "balance": [100.0, 90.0, 20.0, 80.0, 110.0, 120.0, 50.0, 40.0],
                "trans_type": ["C", "D", "D", "C", "C", "D", "C", "D"],
                "operation": ["CIC", "CCW", "CCW", "CIC", "CIC", "CCW", "CIC", "CCW"],
                "category": [None, "HH", "HH", None, None, "HH", None, "HH"],
            }
        )
        self.manifest = pd.DataFrame(
            {"account_id": [1, 2], "split": ["train", "valid"]}
        )

    def test_forward_tasks_have_complete_windows_and_future_only_targets(self) -> None:
        stress = build_cashflow_stress_table(
            self.events,
            self.manifest,
            low_balance_threshold=30.0,
            horizon_days=2,
            min_history_transactions=2,
            cutoff_stride_days=3,
            observation_end="2018-01-12",
        )
        value = build_future_value_table(
            self.events,
            self.manifest,
            horizon_days=2,
            min_history_transactions=2,
            cutoff_stride_days=3,
            observation_end="2018-01-12",
        )

        first_stress = stress.loc[stress["sample_id"].eq("cashflow_stress:account:1:cutoff:2018-01-02")].iloc[0]
        first_value = value.loc[value["sample_id"].eq("future_value:account:1:cutoff:2018-01-02")].iloc[0]
        self.assertEqual(first_stress["target"], 1)
        self.assertEqual(first_stress["future_min_balance"], 20.0)
        self.assertEqual(first_stress["future_transaction_count"], 1)
        self.assertAlmostEqual(first_value["future_transaction_volume"], 10.0)
        self.assertAlmostEqual(first_value["target"], np.log1p(10.0))
        self.assertTrue((stress["cutoff_day"] <= pd.Timestamp("2018-01-10")).all())
        self.assertTrue((stress["cutoff_time"] > stress["cutoff_day"]).all())

    def test_low_balance_threshold_is_train_fitted(self) -> None:
        threshold = fit_low_balance_threshold(self.events.loc[self.events["account_id"].eq(1)], quantile=0.5)
        self.assertEqual(threshold, 95.0)

    def test_cashflow_task_excludes_accounts_already_in_stress_at_cutoff(self) -> None:
        already_stressed = pd.DataFrame(
            {
                "trans_id": [1, 2, 3],
                "account_id": [9, 9, 9],
                "trans_date": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-05"]),
                "amount": [5.0, 5.0, 5.0],
                "balance": [25.0, 20.0, 15.0],
            }
        )
        task = build_cashflow_stress_table(
            already_stressed,
            pd.DataFrame({"account_id": [9], "split": ["train"]}),
            low_balance_threshold=30.0,
            horizon_days=2,
            min_history_transactions=1,
            cutoff_stride_days=1,
            observation_end="2018-01-07",
        )
        self.assertTrue(task.empty)


class TabularFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = pd.DataFrame(
            {
                "trans_id": [1, 2, 3, 4],
                "account_id": [1, 1, 1, 2],
                "trans_date": pd.to_datetime(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-02"]),
                "amount": [10.0, 20.0, 999.0, 30.0],
                "balance": [100.0, 80.0, -200.0, 50.0],
                "trans_type": ["C", "D", "D", "C"],
                "operation": ["CIC", "CCW", "FUTURE_ONLY", "CIC"],
                "category": [None, "HH", "ZZ", None],
            }
        )
        self.profiles = pd.DataFrame(
            {
                "account_id": [1, 2],
                "create_date": pd.to_datetime(["2017-01-01", "2017-06-01"]),
            }
        )

    def test_features_exclude_post_cutoff_rows_and_schema_is_train_only(self) -> None:
        tasks = pd.DataFrame(
            {
                "sample_id": ["train:1", "valid:2"],
                "account_id": [1, 2],
                "cutoff_time": pd.to_datetime(["2018-01-02 23:59:59.999999", "2018-01-02 23:59:59.999999"]),
            }
        )
        features = build_account_history_features(self.events, self.profiles, tasks)
        train_features = features.loc[["train:1"]]
        valid_features = features.loc[["valid:2"]]
        schema = fit_tabular_feature_schema(train_features)
        transformed_valid = schema.transform(valid_features)

        self.assertEqual(features.loc["train:1", "amount_total_abs"], 30.0)
        self.assertEqual(features.loc["train:1", "balance_last"], 80.0)
        self.assertNotIn("operation_share::FUTURE_ONLY", schema.feature_names)
        self.assertEqual(tuple(transformed_valid.columns), schema.feature_names)
        self.assertFalse(transformed_valid.isna().any().any())


@unittest.skipUnless(find_spec("sklearn") is not None, "scikit-learn is not installed")
class EvaluationFrameworkTest(unittest.TestCase):
    def test_binary_and_regression_benchmarks_have_finite_reports(self) -> None:
        train_features = pd.DataFrame({"x": [-3, -2, -1, -0.5, 0.5, 1, 2, 3], "y": [0, 0, 0, 0, 1, 1, 1, 1]})
        valid_features = pd.DataFrame({"x": [-1.5, -0.5, 0.5, 1.5], "y": [0, 0, 1, 1]})
        test_features = pd.DataFrame({"x": [-2.5, -0.25, 0.25, 2.5], "y": [0, 0, 1, 1]})
        train_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        valid_labels = np.array([0, 0, 1, 1])
        test_labels = np.array([0, 0, 1, 1])

        binary = fit_binary_logistic_benchmark(
            train_features, train_labels, valid_features, valid_labels, test_features, test_labels
        )
        probabilities = binary.estimator.predict_proba(test_features)[:, 1]
        calibrated = fit_platt_probability_calibrator(valid_labels, binary.estimator.predict_proba(valid_features)[:, 1])
        regression = fit_ridge_regression_benchmark(
            train_features,
            train_features["x"].to_numpy(),
            valid_features,
            valid_features["x"].to_numpy(),
            test_features,
            test_features["x"].to_numpy(),
        )

        self.assertGreater(binary.test_metrics["average_precision"], 0.9)
        self.assertTrue(np.isfinite(calibrated.transform(probabilities)).all())
        self.assertLess(regression.test_metrics["mae"], 0.1)
        self.assertGreater(binary_classification_metrics(test_labels, probabilities)["roc_auc"], 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
