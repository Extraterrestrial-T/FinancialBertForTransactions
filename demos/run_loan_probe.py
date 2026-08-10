"""Evaluate a pretrained PRAGMA-lite encoder on observed loan repayment trouble.

The model receives account/profile information strictly before a loan grant
date.  ``B`` (finished unpaid) and ``D`` (running and in debt) are positive
labels; the loan status never enters the encoder.  This is an exploratory
frozen-embedding transfer probe, not a deployable credit-scoring claim.

Example in Colab after the MLM notebook has produced ``best.pt``::

    python demos/run_loan_probe.py \
        --checkpoint /content/drive/MyDrive/FinancialBertForTransactions/checkpoints/pragma_lite_mlm/best.pt
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finBERTlitemodules import (  # noqa: E402
    EventMLMDemoModel,
    FinBERTLiteCzechDataset,
    TransformerConfig,
    build_loan_outcome_table,
    collate_account_records,
    load_tokenizer_bundle,
    read_loan_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-pre-grant-transactions", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="metrics JSON path; defaults beside the checkpoint",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.min_pre_grant_transactions < 1:
        parser.error("batch-size and min-pre-grant-transactions must be positive")
    return args


def move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def forward_from_batch(model: EventMLMDemoModel, batch: dict[str, Any]):
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


def extract_embeddings(
    model: EventMLMDemoModel,
    dataset: FinBERTLiteCzechDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned account IDs, frozen account embeddings, and labels."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_account_records,
    )
    account_ids: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_tensors(raw_batch, device)
            output = forward_from_batch(model, batch)
            account_ids.append(raw_batch["account_ids"].numpy())
            embeddings.append(output.account_embedding.cpu().numpy())
            labels.append(raw_batch["targets"].numpy())
    return np.concatenate(account_ids), np.concatenate(embeddings), np.concatenate(labels)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        roc_auc_score,
    )

    if np.unique(labels).size != 2:
        raise ValueError("both target classes are required to compute probe metrics")
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "balanced_accuracy_at_0.5": float(
            balanced_accuracy_score(labels, probabilities >= 0.5)
        ),
        "positive_rate": float(labels.mean()),
        "n_examples": int(labels.size),
        "n_positive": int(labels.sum()),
    }


def bootstrap_auc_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int = 23,
    samples: int = 2_000,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval, exposing small-test uncertainty."""
    from sklearn.metrics import roc_auc_score

    generator = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(samples):
        indices = generator.integers(0, labels.size, size=labels.size)
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size == 2:
            scores.append(float(roc_auc_score(sampled_labels, probabilities[indices])))
    if not scores:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(scores, [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise SystemExit("Install scikit-learn first: pip install scikit-learn") from error

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TransformerConfig(**checkpoint["model_config"])
    model = EventMLMDemoModel(checkpoint["vocabulary_size"], config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    processed_directory = ROOT / "data" / "processed" / "czech_bank"
    source_directory = ROOT / "financial_db_Teradata"
    split_manifest = pd.read_parquet(processed_directory / "account_split_manifest.parquet")
    events_by_split = {
        split: processed_directory / f"events_{split}.parquet"
        for split in ("train", "valid", "test")
    }
    profiles_by_split = {
        split: processed_directory / f"profile_{split}.parquet"
        for split in ("train", "valid", "test")
    }
    for required_path in (*events_by_split.values(), *profiles_by_split.values(), source_directory):
        if not required_path.exists():
            raise FileNotFoundError(f"missing required Czech data: {required_path}")

    all_events = pd.concat(
        [pd.read_parquet(path) for path in events_by_split.values()], ignore_index=True
    )
    task_table = build_loan_outcome_table(
        all_events,
        read_loan_outcomes(source_directory),
        split_manifest,
        min_pre_grant_transactions=args.min_pre_grant_transactions,
    )
    tokenizers = load_tokenizer_bundle(checkpoint_path.parent / "tokenizers")
    max_events = int(checkpoint.get("max_events", 64))

    split_embeddings: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in ("train", "valid", "test"):
        split_task = task_table.loc[task_table["split"].eq(split)]
        fixed_cutoffs = dict(zip(split_task["account_id"], split_task["cutoff_time"], strict=True))
        targets = dict(zip(split_task["account_id"], split_task["target"], strict=True))
        dataset = FinBERTLiteCzechDataset(
            events_by_split[split],
            profiles_by_split[split],
            source_directory,
            tokenizers,
            max_events=max_events,
            random_cutoff=False,
            fixed_cutoffs_by_account=fixed_cutoffs,
            targets_by_account=targets,
        )
        split_embeddings[split] = extract_embeddings(
            model, dataset, batch_size=args.batch_size, device=device
        )
        _, _, labels = split_embeddings[split]
        print(f"{split}: n={labels.size}, observed trouble={int(labels.sum())}")

    _, train_embeddings, train_labels = split_embeddings["train"]
    _, valid_embeddings, valid_labels = split_embeddings["valid"]
    test_account_ids, test_embeddings, test_labels = split_embeddings["test"]

    # Tune the linear-probe regularisation only on validation, then report the
    # test result once. The pretrained encoder remains completely frozen.
    candidates = (0.01, 0.1, 1.0, 10.0)
    validation_scores: dict[float, float] = {}
    for regularisation in candidates:
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularisation,
                class_weight="balanced",
                max_iter=2_000,
                random_state=17,
            ),
        )
        probe.fit(train_embeddings, train_labels)
        validation_scores[regularisation] = float(
            average_precision_score(valid_labels, probe.predict_proba(valid_embeddings)[:, 1])
        )
    selected_c = max(validation_scores, key=validation_scores.get)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=selected_c,
            class_weight="balanced",
            max_iter=2_000,
            random_state=17,
        ),
    )
    probe.fit(train_embeddings, train_labels)
    validation_probabilities = probe.predict_proba(valid_embeddings)[:, 1]
    test_probabilities = probe.predict_proba(test_embeddings)[:, 1]
    validation_metrics = classification_metrics(valid_labels, validation_probabilities)
    test_metrics = classification_metrics(test_labels, test_probabilities)
    test_auc_interval = bootstrap_auc_interval(test_labels, test_probabilities)

    result = {
        "task": "observed_repayment_trouble_before_loan_grant",
        "positive_statuses": ["B", "D"],
        "min_pre_grant_transactions": args.min_pre_grant_transactions,
        "checkpoint": str(checkpoint_path),
        "selected_logistic_c": selected_c,
        "validation_average_precision_by_c": validation_scores,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_roc_auc_bootstrap_95_interval": test_auc_interval,
        "test_account_ids": test_account_ids.tolist(),
        "protocol_caveat": (
            "The MLM checkpoint was not horizon-restricted to loan grant dates. "
            "Treat this as an exploratory frozen-embedding transfer probe, not "
            "a prospective credit-scoring estimate."
        ),
    }
    output_path = args.output or checkpoint_path.parent / "loan_probe_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"selected C={selected_c}; validation={validation_metrics}")
    print(f"test={test_metrics}")
    print(f"test ROC-AUC bootstrap 95% interval={test_auc_interval}")
    print(f"saved metrics: {output_path}")
    print("Caveat: only eight positive labels are in the current test split; use this result descriptively.")


if __name__ == "__main__":
    main()
