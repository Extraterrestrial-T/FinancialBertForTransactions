"""PRAGMA-lite: a compact foundation-model experiment for financial events."""

from .data import (
    EventTokenizer,
    FinBERTLiteCzechDataset,
    LifelongEventTokenizer,
    ProfileTokenizer,
    TokenizerBundle,
    apply_value_mlm_mask,
    collate_account_records,
    fit_tokenizer_bundle,
    load_tokenizer_bundle,
    save_tokenizer_bundle,
)
from .models import (
    EventEncoder,
    EventEncoderOutput,
    EventMLMDemoModel,
    EventMLMOutput,
    FeedForwardNetwork,
    HistoryEncoder,
    HistoryEncoderOutput,
    MultiHeadAttention,
    ProfileEncoder,
    ProfileEncoderOutput,
    TransformerBlock,
    TransformerConfig,
    TransformerModel,
    apply_rotary_embedding,
)
from .inference import (
    AccountPrediction,
    AccountTaskPredictor,
    PreparedAccountSample,
    load_processed_czech_account_sample,
)
from .demo import (
    DemoSnapshot,
    DemoTaskSpec,
    load_demo_account_index,
    load_demo_snapshot,
    task_spec,
    transaction_bucket_label,
    valid_demo_cutoffs,
)
from .tasks import (
    PROBLEM_LOAN_STATUSES,
    BinaryLogisticBenchmark,
    PlattProbabilityCalibrator,
    RidgeRegressionBenchmark,
    TabularFeatureSchema,
    binary_classification_metrics,
    bootstrap_roc_auc_interval,
    clustered_binary_bootstrap_intervals,
    clustered_regression_bootstrap_intervals,
    build_account_history_features,
    build_cashflow_stress_table,
    build_future_value_table,
    build_loan_outcome_table,
    fit_binary_logistic_benchmark,
    fit_low_balance_threshold,
    fit_platt_probability_calibrator,
    fit_ridge_regression_benchmark,
    fit_tabular_feature_schema,
    prevalence_baseline_probabilities,
    read_loan_outcomes,
    regression_metrics,
)
from .training import (
    AccountTaskModel,
    FineTuneConfig,
    HistoryLoRAState,
    LoRAConfig,
    LoRALinear,
    PretrainingConfig,
    adapter_state_dict,
    inject_history_lora,
    load_lora_task_model,
    run_lora_finetuning,
    run_pretraining,
)

__all__ = [name for name in globals() if not name.startswith("_")]
