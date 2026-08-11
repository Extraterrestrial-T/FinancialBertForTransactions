"""Leakage-safe downstream task construction, baselines, and metrics."""

from .definitions import (
    PROBLEM_LOAN_STATUSES,
    build_cashflow_stress_table,
    build_future_value_table,
    build_loan_outcome_table,
    fit_low_balance_threshold,
    read_loan_outcomes,
)
from .evaluation import (
    BinaryLogisticBenchmark,
    PlattProbabilityCalibrator,
    RidgeRegressionBenchmark,
    binary_classification_metrics,
    bootstrap_roc_auc_interval,
    clustered_binary_bootstrap_intervals,
    clustered_regression_bootstrap_intervals,
    fit_binary_logistic_benchmark,
    fit_platt_probability_calibrator,
    fit_ridge_regression_benchmark,
    prevalence_baseline_probabilities,
    regression_metrics,
)
from .features import TabularFeatureSchema, build_account_history_features, fit_tabular_feature_schema

__all__ = [
    "PROBLEM_LOAN_STATUSES",
    "read_loan_outcomes",
    "build_loan_outcome_table",
    "fit_low_balance_threshold",
    "build_cashflow_stress_table",
    "build_future_value_table",
    "TabularFeatureSchema",
    "build_account_history_features",
    "fit_tabular_feature_schema",
    "BinaryLogisticBenchmark",
    "binary_classification_metrics",
    "bootstrap_roc_auc_interval",
    "clustered_binary_bootstrap_intervals",
    "clustered_regression_bootstrap_intervals",
    "prevalence_baseline_probabilities",
    "fit_binary_logistic_benchmark",
    "PlattProbabilityCalibrator",
    "fit_platt_probability_calibrator",
    "RidgeRegressionBenchmark",
    "regression_metrics",
    "fit_ridge_regression_benchmark",
]
