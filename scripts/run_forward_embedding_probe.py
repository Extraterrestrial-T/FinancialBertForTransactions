"""Evaluate frozen PRAGMA-lite account embeddings on forward-looking tasks.

The encoder is frozen.  A logistic (cash-flow stress) or Ridge (future-value)
head is fitted on account-disjoint training examples and selected on validation
only.  This is the transfer test to compare with the corresponding tabular
baseline before considering any fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pragma_lite import (  # noqa: E402
    EventMLMDemoModel,
    FinBERTLiteCzechDataset,
    TransformerConfig,
    build_cashflow_stress_table,
    build_future_value_table,
    clustered_binary_bootstrap_intervals,
    clustered_regression_bootstrap_intervals,
    collate_account_records,
    fit_binary_logistic_benchmark,
    fit_low_balance_threshold,
    fit_ridge_regression_benchmark,
    load_tokenizer_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("cashflow_stress", "future_value"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task-table",
        type=Path,
        default=None,
        help="optional cached task table from run_forward_task_baseline.py",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cutoff-stride-days", type=int, default=180)
    parser.add_argument("--min-history-transactions", type=int, default=20)
    parser.add_argument("--cashflow-horizon-days", type=int, default=60)
    parser.add_argument("--low-balance-quantile", type=float, default=0.10)
    parser.add_argument("--value-horizon-days", type=int, default=180)
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cutoff_stride_days < 1 or args.min_history_transactions < 1 or args.bootstrap_samples < 1:
        parser.error("batch size, cutoff stride, minimum history, and bootstrap-samples must be positive")
    return args


def _move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def _forward_from_batch(model: EventMLMDemoModel, batch: dict[str, Any]):
    return model(
        batch["event_key_ids"],
        batch["event_value_ids"],
        batch["event_mask"],
        batch["profile_key_ids"],
        batch["profile_value_ids"],
        batch["profile_mask"],
        batch["profile_rope_time"],
        batch["history_rope_time"],
        batch["calendar_features"],
    )


def _extract_embeddings(
    model: EventMLMDemoModel,
    dataset: FinBERTLiteCzechDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_account_records,
    )
    sample_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    account_ids: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_tensors(raw_batch, device)
            output = _forward_from_batch(model, batch)
            sample_ids.extend(raw_batch["sample_ids"])
            embeddings.append(output.account_embedding.cpu().numpy())
            labels.append(raw_batch["targets"].cpu().numpy())
            account_ids.append(raw_batch["account_ids"].cpu().numpy())
    return sample_ids, np.concatenate(embeddings), np.concatenate(labels), np.concatenate(account_ids)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    processed = ROOT / "data" / "processed" / "czech_bank"
    lifelong_events_path = processed / "lifelong_events.parquet"
    paths = {
        split: {
            "events": processed / f"events_{split}.parquet",
            "profiles": processed / f"profile_{split}.parquet",
        }
        for split in ("train", "valid", "test")
    }
    for path in [lifelong_events_path, processed / "account_split_manifest.parquet", *(path for values in paths.values() for path in values.values())]:
        if not path.exists():
            raise FileNotFoundError(f"missing required data: {path}")

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = EventMLMDemoModel(
        checkpoint["vocabulary_size"], TransformerConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizers = load_tokenizer_bundle(checkpoint_path.parent / "tokenizers")
    max_events = int(checkpoint.get("max_events", 64))

    events_by_split = {split: pd.read_parquet(values["events"]) for split, values in paths.items()}
    if args.task_table is not None:
        if not args.task_table.exists():
            raise FileNotFoundError(f"cached task table not found: {args.task_table}")
        task_table = pd.read_parquet(args.task_table)
        required_task_columns = {
            "sample_id",
            "account_id",
            "cutoff_time",
            "window_end",
            "split",
            "target",
        }
        if missing := required_task_columns.difference(task_table.columns):
            raise ValueError(f"cached task table is missing columns: {sorted(missing)}")
        if set(task_table["split"].unique()).difference({"train", "valid", "test"}):
            raise ValueError("cached task table has unexpected split values")
        if not task_table["sample_id"].astype(str).str.startswith(f"{args.task}:").all():
            raise ValueError("cached task table does not match the selected task")
        if args.task == "cashflow_stress" and "low_balance_threshold" not in task_table:
            raise ValueError("cached cash-flow task table is missing low_balance_threshold")
        observation_end = pd.to_datetime(task_table["window_end"]).max()
        threshold = (
            float(task_table["low_balance_threshold"].iloc[0])
            if args.task == "cashflow_stress"
            else None
        )
    else:
        all_events = pd.concat(events_by_split.values(), ignore_index=True)
        manifest = pd.read_parquet(processed / "account_split_manifest.parquet")
        observation_end = all_events["trans_date"].max()
        if args.task == "cashflow_stress":
            threshold = fit_low_balance_threshold(events_by_split["train"], quantile=args.low_balance_quantile)
            task_table = build_cashflow_stress_table(
                all_events, manifest, low_balance_threshold=threshold,
                horizon_days=args.cashflow_horizon_days,
                min_history_transactions=args.min_history_transactions,
                cutoff_stride_days=args.cutoff_stride_days,
                observation_end=observation_end,
            )
        else:
            threshold = None
            task_table = build_future_value_table(
                all_events, manifest,
                horizon_days=args.value_horizon_days,
                min_history_transactions=args.min_history_transactions,
                cutoff_stride_days=args.cutoff_stride_days,
                observation_end=observation_end,
            )

    embeddings: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    account_ids: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        split_rows = task_table.loc[task_table["split"].eq(split)]
        dataset = FinBERTLiteCzechDataset(
            paths[split]["events"],
            paths[split]["profiles"],
            lifelong_events_path,
            tokenizers,
            max_events=max_events,
            random_cutoff=False,
            task_table=split_rows,
        )
        sample_ids, vectors, labels, accounts = _extract_embeddings(
            model, dataset, batch_size=args.batch_size, device=device
        )
        embeddings[split] = pd.DataFrame(vectors, index=sample_ids)
        targets[split] = labels.astype(np.int64 if args.task == "cashflow_stress" else np.float64)
        account_ids[split] = accounts.astype(np.int64)
        print(f"{split}: n={len(labels)}")

    report: dict[str, Any] = {
        "task": args.task,
        "checkpoint": str(checkpoint_path),
        "encoder": "frozen EventMLMDemoModel account_embedding",
        "task_definition": {
            "cutoff_stride_days": args.cutoff_stride_days,
            "min_history_transactions": args.min_history_transactions,
            "observation_end": str(pd.Timestamp(observation_end).date()),
            "account_disjoint_splits": True,
            "full_future_windows_only": True,
            "cached_task_table": str(args.task_table) if args.task_table is not None else None,
        },
        "sample_counts": {split: int(len(targets[split])) for split in targets},
        "account_cluster_bootstrap_samples": args.bootstrap_samples,
        "protocol_caveat": (
            "The MLM checkpoint was not horizon-restricted to every downstream cutoff. "
            "Treat this as an exploratory frozen-embedding transfer result, not a "
            "fully prospective deployment estimate."
        ),
    }
    if args.task == "cashflow_stress":
        benchmark = fit_binary_logistic_benchmark(
            embeddings["train"], targets["train"], embeddings["valid"], targets["valid"],
            embeddings["test"], targets["test"],
        )
        probabilities = {
            split: benchmark.estimator.predict_proba(embeddings[split])[:, 1]
            for split in ("valid", "test")
        }
        report.update(
            {
                "low_balance_threshold": threshold,
                "horizon_days": args.cashflow_horizon_days,
                "frozen_embedding_logistic_probe": {
                    **benchmark.report(),
                    "validation_account_cluster_bootstrap_95_intervals": (
                        clustered_binary_bootstrap_intervals(
                            targets["valid"], probabilities["valid"], account_ids["valid"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                    "test_account_cluster_bootstrap_95_intervals": (
                        clustered_binary_bootstrap_intervals(
                            targets["test"], probabilities["test"], account_ids["test"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                },
            }
        )
    else:
        benchmark = fit_ridge_regression_benchmark(
            embeddings["train"], targets["train"], embeddings["valid"], targets["valid"],
            embeddings["test"], targets["test"],
        )
        predictions = {
            split: benchmark.estimator.predict(embeddings[split]) for split in ("valid", "test")
        }
        report.update(
            {
                "horizon_days": args.value_horizon_days,
                "target": "log1p(sum(abs(amount)) in future window)",
                "frozen_embedding_ridge_probe": {
                    **benchmark.report(),
                    "validation_account_cluster_bootstrap_95_intervals": (
                        clustered_regression_bootstrap_intervals(
                            targets["valid"], predictions["valid"], account_ids["valid"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                    "test_account_cluster_bootstrap_95_intervals": (
                        clustered_regression_bootstrap_intervals(
                            targets["test"], predictions["test"], account_ids["test"],
                            samples=args.bootstrap_samples,
                        )
                    ),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved frozen-embedding report: {args.output}")


if __name__ == "__main__":
    main()
