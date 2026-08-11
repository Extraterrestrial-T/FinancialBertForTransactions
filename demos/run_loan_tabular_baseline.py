"""Run the leakage-safe tabular baseline for observed loan repayment trouble.

This is the comparator for ``demos/run_loan_probe.py``.  It uses only
hand-built historical aggregates at the same pre-grant cutoff, and fits its
feature schema and model choices on training data only.

Example::

    python demos/run_loan_tabular_baseline.py \
      --output /content/drive/MyDrive/FinancialBertForTransactions/checkpoints/pragma_lite_mlm/loan_tabular_baseline_metrics.json \
      --frozen-probe-metrics /content/drive/MyDrive/FinancialBertForTransactions/checkpoints/pragma_lite_mlm/loan_probe_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finBERTlitemodules import (  # noqa: E402
    binary_classification_metrics,
    bootstrap_roc_auc_interval,
    build_account_history_features,
    build_loan_outcome_table,
    fit_binary_logistic_benchmark,
    fit_tabular_feature_schema,
    prevalence_baseline_probabilities,
    read_loan_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-probe-metrics", type=Path, default=None)
    parser.add_argument("--min-pre-grant-transactions", type=int, default=20)
    args = parser.parse_args()
    if args.min_pre_grant_transactions < 1:
        parser.error("min-pre-grant-transactions must be positive")
    return args


def task_labels(task_rows: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    """Align labels to sorted feature rows via stable task sample IDs."""
    labels = task_rows.set_index("sample_id")["target"]
    if not features.index.isin(labels.index).all():
        raise ValueError("feature rows are not all represented in task labels")
    return labels.loc[features.index].to_numpy(dtype=np.int64)


def main() -> None:
    args = parse_args()
    processed_directory = ROOT / "data" / "processed" / "czech_bank"
    source_directory = ROOT / "financial_db_Teradata"
    paths = {
        split: {
            "events": processed_directory / f"events_{split}.parquet",
            "profiles": processed_directory / f"profile_{split}.parquet",
        }
        for split in ("train", "valid", "test")
    }
    required = [source_directory, processed_directory / "account_split_manifest.parquet"]
    required.extend(path for split_paths in paths.values() for path in split_paths.values())
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"missing required Czech data: {path}")

    events_by_split = {split: pd.read_parquet(values["events"]) for split, values in paths.items()}
    profiles_by_split = {split: pd.read_parquet(values["profiles"]) for split, values in paths.items()}
    split_manifest = pd.read_parquet(processed_directory / "account_split_manifest.parquet")
    task_table = build_loan_outcome_table(
        pd.concat(events_by_split.values(), ignore_index=True),
        read_loan_outcomes(source_directory),
        split_manifest,
        min_pre_grant_transactions=args.min_pre_grant_transactions,
    )

    raw_features: dict[str, pd.DataFrame] = {}
    labels: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        task_rows = task_table.loc[task_table["split"].eq(split)]
        raw_features[split] = build_account_history_features(
            events_by_split[split], profiles_by_split[split], task_rows
        )
        labels[split] = task_labels(task_rows, raw_features[split])
        print(f"{split}: n={labels[split].size}, observed trouble={int(labels[split].sum())}")

    schema = fit_tabular_feature_schema(raw_features["train"])
    features = {split: schema.transform(raw_features[split]) for split in raw_features}
    prevalence_valid = prevalence_baseline_probabilities(labels["train"], labels["valid"].size)
    prevalence_test = prevalence_baseline_probabilities(labels["train"], labels["test"].size)
    logistic = fit_binary_logistic_benchmark(
        features["train"],
        labels["train"],
        features["valid"],
        labels["valid"],
        features["test"],
        labels["test"],
    )

    report: dict[str, Any] = {
        "task": "observed_repayment_trouble_before_loan_grant",
        "min_pre_grant_transactions": args.min_pre_grant_transactions,
        "feature_count": len(schema.feature_names),
        "feature_names": list(schema.feature_names),
        "sample_counts": {
            split: {"n_examples": int(labels[split].size), "n_positive": int(labels[split].sum())}
            for split in labels
        },
        "prevalence_baseline": {
            "train_positive_rate": float(labels["train"].mean()),
            "validation_metrics": binary_classification_metrics(labels["valid"], prevalence_valid),
            "test_metrics": binary_classification_metrics(labels["test"], prevalence_test),
            "test_roc_auc_bootstrap_95_interval": bootstrap_roc_auc_interval(
                labels["test"], prevalence_test
            ),
        },
        "tabular_logistic_baseline": logistic.report(),
        "protocol": {
            "input_cutoff": "one microsecond before loan grant date",
            "same_day_transactions_excluded": True,
            "loan_granted_lifelong_event_excluded": True,
            "account_disjoint_splits": True,
            "feature_schema_fit_on_train_only": True,
            "hyperparameter_selected_on_validation_average_precision": True,
            "test_set_used_only_for_final_report": True,
        },
    }
    if args.frozen_probe_metrics is not None:
        if not args.frozen_probe_metrics.exists():
            raise FileNotFoundError(f"frozen probe metrics not found: {args.frozen_probe_metrics}")
        frozen_probe = json.loads(args.frozen_probe_metrics.read_text(encoding="utf-8"))
        if frozen_probe.get("test_metrics", {}).get("n_examples") != int(labels["test"].size):
            raise ValueError("frozen probe and tabular baseline used different test samples")
        report["frozen_embedding_probe"] = frozen_probe

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    schema.save(args.output.with_name("loan_tabular_feature_schema.json"))
    print(f"tabular validation={logistic.validation_metrics}")
    print(f"tabular test={logistic.test_metrics}")
    print(f"saved benchmark: {args.output}")


if __name__ == "__main__":
    main()
