"""Run a leakage-safe tabular baseline for cash-flow stress or future value.

The cash-flow task predicts whether an account reaches the train-defined low
balance threshold during the next 60 days.  The future-value task predicts
``log1p`` future absolute transaction volume over the next 180 days.  Both
generate several dated snapshots per account while preserving account splits.

Example::

    python scripts/run_forward_task_baseline.py --task cashflow_stress \
      --output /content/drive/MyDrive/FinancialBertForTransactions/checkpoints/pragma_lite_mlm/cashflow_tabular_baseline.json
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
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pragma_lite import (  # noqa: E402
    binary_classification_metrics,
    bootstrap_roc_auc_interval,
    clustered_binary_bootstrap_intervals,
    clustered_regression_bootstrap_intervals,
    build_account_history_features,
    build_cashflow_stress_table,
    build_future_value_table,
    fit_binary_logistic_benchmark,
    fit_low_balance_threshold,
    fit_ridge_regression_benchmark,
    fit_tabular_feature_schema,
    prevalence_baseline_probabilities,
    regression_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("cashflow_stress", "future_value"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-stride-days", type=int, default=180)
    parser.add_argument("--min-history-transactions", type=int, default=20)
    parser.add_argument("--cashflow-horizon-days", type=int, default=60)
    parser.add_argument("--low-balance-quantile", type=float, default=0.10)
    parser.add_argument("--value-horizon-days", type=int, default=180)
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    args = parser.parse_args()
    if args.cutoff_stride_days < 1 or args.min_history_transactions < 1 or args.bootstrap_samples < 1:
        parser.error("cutoff-stride-days, min-history-transactions, and bootstrap-samples must be positive")
    return args


def _labels_aligned(task_rows: pd.DataFrame, features: pd.DataFrame, *, dtype: str) -> np.ndarray:
    targets = task_rows.set_index("sample_id")["target"]
    if not features.index.isin(targets.index).all():
        raise ValueError("features and task labels are misaligned")
    return targets.loc[features.index].to_numpy(dtype=dtype)


def _accounts_aligned(task_rows: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    accounts = task_rows.set_index("sample_id")["account_id"]
    if not features.index.isin(accounts.index).all():
        raise ValueError("features and account IDs are misaligned")
    return accounts.loc[features.index].to_numpy(dtype="int64")


def main() -> None:
    args = parse_args()
    processed = ROOT / "data" / "processed" / "czech_bank"
    source_paths = {
        split: {
            "events": processed / f"events_{split}.parquet",
            "profiles": processed / f"profile_{split}.parquet",
        }
        for split in ("train", "valid", "test")
    }
    required = [processed / "account_split_manifest.parquet"]
    required.extend(path for values in source_paths.values() for path in values.values())
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"missing required Czech data: {path}")

    events_by_split = {split: pd.read_parquet(paths["events"]) for split, paths in source_paths.items()}
    profiles_by_split = {split: pd.read_parquet(paths["profiles"]) for split, paths in source_paths.items()}
    all_events = pd.concat(events_by_split.values(), ignore_index=True)
    manifest = pd.read_parquet(processed / "account_split_manifest.parquet")
    observation_end = all_events["trans_date"].max()

    if args.task == "cashflow_stress":
        threshold = fit_low_balance_threshold(
            events_by_split["train"], quantile=args.low_balance_quantile
        )
        task_table = build_cashflow_stress_table(
            all_events,
            manifest,
            low_balance_threshold=threshold,
            horizon_days=args.cashflow_horizon_days,
            min_history_transactions=args.min_history_transactions,
            cutoff_stride_days=args.cutoff_stride_days,
            observation_end=observation_end,
        )
    else:
        threshold = None
        task_table = build_future_value_table(
            all_events,
            manifest,
            horizon_days=args.value_horizon_days,
            min_history_transactions=args.min_history_transactions,
            cutoff_stride_days=args.cutoff_stride_days,
            observation_end=observation_end,
        )
    task_table_path = args.output.with_name(f"{args.task}_task_table.parquet")
    task_table_path.parent.mkdir(parents=True, exist_ok=True)
    task_table.to_parquet(task_table_path, index=False)

    raw_features: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    account_ids: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        split_rows = task_table.loc[task_table["split"].eq(split)]
        raw_features[split] = build_account_history_features(
            events_by_split[split], profiles_by_split[split], split_rows
        )
        targets[split] = _labels_aligned(
            split_rows,
            raw_features[split],
            dtype="int64" if args.task == "cashflow_stress" else "float64",
        )
        account_ids[split] = _accounts_aligned(split_rows, raw_features[split])
        summary = (
            f"{split}: n={len(raw_features[split])}, positive={int(targets[split].sum())}"
            if args.task == "cashflow_stress"
            else f"{split}: n={len(raw_features[split])}, target_mean={targets[split].mean():.4f}"
        )
        print(summary)

    schema = fit_tabular_feature_schema(raw_features["train"])
    features = {split: schema.transform(raw_features[split]) for split in raw_features}
    common_report: dict[str, Any] = {
        "task": args.task,
        "task_definition": {
            "cutoff_stride_days": args.cutoff_stride_days,
            "min_history_transactions": args.min_history_transactions,
            "observation_end": str(pd.Timestamp(observation_end).date()),
            "account_disjoint_splits": True,
            "full_future_windows_only": True,
            "feature_schema_fit_on_train_only": True,
            "task_table_path": str(task_table_path),
        },
        "feature_count": len(schema.feature_names),
        "sample_counts": {split: int(len(features[split])) for split in features},
        "account_cluster_bootstrap_samples": args.bootstrap_samples,
    }
    if args.task == "cashflow_stress":
        prevalence_valid = prevalence_baseline_probabilities(targets["train"], len(targets["valid"]))
        prevalence_test = prevalence_baseline_probabilities(targets["train"], len(targets["test"]))
        benchmark = fit_binary_logistic_benchmark(
            features["train"], targets["train"], features["valid"], targets["valid"],
            features["test"], targets["test"],
        )
        tabular_probabilities = {
            split: benchmark.estimator.predict_proba(features[split])[:, 1]
            for split in ("valid", "test")
        }
        common_report.update(
            {
                "low_balance_threshold": threshold,
                "horizon_days": args.cashflow_horizon_days,
                "prevalence_baseline": {
                    "validation_metrics": binary_classification_metrics(targets["valid"], prevalence_valid),
                    "test_metrics": binary_classification_metrics(targets["test"], prevalence_test),
                    "test_roc_auc_bootstrap_95_interval": bootstrap_roc_auc_interval(
                        targets["test"], prevalence_test
                    ),
                    "validation_account_cluster_bootstrap_95_intervals": (
                        clustered_binary_bootstrap_intervals(
                            targets["valid"], prevalence_valid, account_ids["valid"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                    "test_account_cluster_bootstrap_95_intervals": (
                        clustered_binary_bootstrap_intervals(
                            targets["test"], prevalence_test, account_ids["test"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                },
                "tabular_logistic_baseline": {
                    **benchmark.report(),
                    "validation_account_cluster_bootstrap_95_intervals": (
                        clustered_binary_bootstrap_intervals(
                            targets["valid"], tabular_probabilities["valid"], account_ids["valid"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                    "test_account_cluster_bootstrap_95_intervals": (
                        clustered_binary_bootstrap_intervals(
                            targets["test"], tabular_probabilities["test"], account_ids["test"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                },
            }
        )
    else:
        benchmark = fit_ridge_regression_benchmark(
            features["train"], targets["train"], features["valid"], targets["valid"],
            features["test"], targets["test"],
        )
        train_mean = float(targets["train"].mean())
        tabular_predictions = {
            split: benchmark.estimator.predict(features[split]) for split in ("valid", "test")
        }
        common_report.update(
            {
                "horizon_days": args.value_horizon_days,
                "target": "log1p(sum(abs(amount)) in future window)",
                "mean_target_baseline": {
                    "train_target_mean": train_mean,
                    "validation_metrics": regression_metrics(
                        targets["valid"], np.full(len(targets["valid"]), train_mean)
                    ),
                    "test_metrics": regression_metrics(
                        targets["test"], np.full(len(targets["test"]), train_mean)
                    ),
                    "validation_account_cluster_bootstrap_95_intervals": (
                        clustered_regression_bootstrap_intervals(
                            targets["valid"], np.full(len(targets["valid"]), train_mean), account_ids["valid"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                    "test_account_cluster_bootstrap_95_intervals": (
                        clustered_regression_bootstrap_intervals(
                            targets["test"], np.full(len(targets["test"]), train_mean), account_ids["test"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                },
                "tabular_ridge_baseline": {
                    **benchmark.report(),
                    "validation_account_cluster_bootstrap_95_intervals": (
                        clustered_regression_bootstrap_intervals(
                            targets["valid"], tabular_predictions["valid"], account_ids["valid"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                    "test_account_cluster_bootstrap_95_intervals": (
                        clustered_regression_bootstrap_intervals(
                            targets["test"], tabular_predictions["test"], account_ids["test"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(common_report, indent=2), encoding="utf-8")
    schema.save(args.output.with_name(f"{args.task}_tabular_feature_schema.json"))
    print(f"saved task table: {task_table_path}")
    print(f"saved baseline report: {args.output}")


if __name__ == "__main__":
    main()
