"""Public data-pipeline API for the PRAGMA-lite Czech-bank experiment.

Importing from this package is optional; the individual modules remain useful
while you are studying them.  These exports give the future training script a
single, stable entry point.
"""

from .datahandler import (
    FinBERTLiteCzechDataset,
    TokenizerBundle,
    apply_value_mlm_mask,
    collate_account_records,
    fit_tokenizer_bundle,
    load_tokenizer_bundle,
    save_tokenizer_bundle,
)
from .downstream import (
    PROBLEM_LOAN_STATUSES,
    build_loan_outcome_table,
    build_cashflow_stress_table,
    build_future_value_table,
    fit_low_balance_threshold,
    read_loan_outcomes,
)
from .downstream_evaluation import (
    BinaryLogisticBenchmark,
    PlattProbabilityCalibrator,
    RidgeRegressionBenchmark,
    binary_classification_metrics,
    bootstrap_roc_auc_interval,
    fit_binary_logistic_benchmark,
    fit_platt_probability_calibrator,
    fit_ridge_regression_benchmark,
    prevalence_baseline_probabilities,
    regression_metrics,
)
from .downstream_features import (
    TabularFeatureSchema,
    build_account_history_features,
    fit_tabular_feature_schema,
)
from .lifelong_tokenizer import LifelongEventTokenizer
from .profile_tokenizer import ProfileTokenizer
from .tokenizer import EventTokenizer
from .transformer_models import (
    EventEncoder,
    EventEncoderOutput,
    EventMLMDemoModel,
    EventMLMOutput,
    FeedForwardNetwork,
    MultiHeadAttention,
    TransformerBlock,
    TransformerConfig,
    ProfileEncoder,
    ProfileEncoderOutput,
    HistoryEncoder,
    HistoryEncoderOutput,
    TransformerModel,
    apply_rotary_embedding,
)

__all__ = [
    "EventTokenizer",
    "ProfileTokenizer",
    "LifelongEventTokenizer",
    "TokenizerBundle",
    "fit_tokenizer_bundle",
    "save_tokenizer_bundle",
    "load_tokenizer_bundle",
    "FinBERTLiteCzechDataset",
    "collate_account_records",
    "apply_value_mlm_mask",
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
    "prevalence_baseline_probabilities",
    "fit_binary_logistic_benchmark",
    "PlattProbabilityCalibrator",
    "fit_platt_probability_calibrator",
    "RidgeRegressionBenchmark",
    "regression_metrics",
    "fit_ridge_regression_benchmark",
    "TransformerConfig",
    "apply_rotary_embedding",
    "MultiHeadAttention",
    "FeedForwardNetwork",
    "TransformerBlock",
    "EventEncoder",
    "EventEncoderOutput",
    "ProfileEncoder",
    "ProfileEncoderOutput",
    "HistoryEncoder",
    "HistoryEncoderOutput",
    "EventMLMDemoModel",
    "EventMLMOutput",
    "TransformerModel",
]
