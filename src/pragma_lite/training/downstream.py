"""LoRA fine-tuning for leakage-safe forward-looking Czech-bank tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data import FinBERTLiteCzechDataset, collate_account_records, load_tokenizer_bundle
from ..models import EventMLMDemoModel, TransformerConfig
from ..tasks.evaluation import (
    binary_classification_metrics,
    clustered_binary_bootstrap_intervals,
    clustered_regression_bootstrap_intervals,
    regression_metrics,
)
from .lora import LoRAConfig, adapter_state_dict, inject_history_lora, iter_lora_parameters


TaskName = Literal["cashflow_stress", "future_value"]


@dataclass(frozen=True, slots=True)
class FineTuneConfig:
    """Reproducible defaults for small-scale History Encoder LoRA experiments."""

    ranks: tuple[int, ...] = (4, 8, 16)
    alpha: float = 8.0
    lora_dropout: float = 0.0
    head_learning_rate: float = 1e-3
    lora_learning_rate: float = 3e-4
    max_epochs: int = 50
    patience: int = 8
    batch_size: int = 128
    gradient_clip_norm: float = 1.0
    bootstrap_samples: int = 250
    seed: int = 42
    num_workers: int = 2

    def __post_init__(self) -> None:
        if not self.ranks or any(rank < 1 for rank in self.ranks):
            raise ValueError("ranks must contain positive values")
        if self.alpha <= 0.0 or self.head_learning_rate <= 0.0 or self.lora_learning_rate <= 0.0:
            raise ValueError("alpha and learning rates must be positive")
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1 or self.bootstrap_samples < 1:
            raise ValueError("epochs, patience, batch_size, and bootstrap_samples must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")


class AccountTaskModel(nn.Module):
    """Frozen MLM backbone plus a trainable linear account-level task head."""

    def __init__(self, backbone: EventMLMDemoModel, task: TaskName) -> None:
        super().__init__()
        self.backbone = backbone
        self.task = task
        self.head = nn.Linear(backbone.config.d_model, 1)

    def forward_from_batch(self, batch: dict[str, Any]) -> Tensor:
        output = self.backbone(
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
        return self.head(output.account_embedding).squeeze(-1)


def run_lora_finetuning(
    *,
    task: TaskName,
    base_checkpoint_path: str | Path,
    task_table_path: str | Path,
    processed_dir: str | Path,
    output_dir: str | Path,
    config: FineTuneConfig | None = None,
    device: str | torch.device | None = None,
    baseline_report_path: str | Path | None = None,
    frozen_probe_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select a LoRA rank on validation, then run one final held-out test.

    The task table is mandatory so adaptation consumes the exact dated samples
    already used by the tabular baseline and frozen probe.
    """
    config = FineTuneConfig() if config is None else config
    task_table_path = Path(task_table_path).resolve()
    processed_dir = Path(processed_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not task_table_path.exists():
        raise FileNotFoundError(f"task table not found: {task_table_path}")
    task_table = pd.read_parquet(task_table_path)
    _validate_task_table(task_table, task)

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(config.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    datasets, max_events = _build_task_datasets(
        task_table, processed_dir=processed_dir, base_checkpoint_path=base_checkpoint_path
    )
    loaders = _build_loaders(datasets, config=config, device=resolved_device)

    rank_reports: list[dict[str, Any]] = []
    adapter_paths: dict[int, Path] = {}
    for rank in config.ranks:
        result, adapter_path = _train_one_rank(
            task=task,
            rank=rank,
            base_checkpoint_path=base_checkpoint_path,
            train_loader=loaders["train"],
            valid_loader=loaders["valid"],
            output_dir=output_dir,
            config=config,
            device=resolved_device,
            task_table_path=task_table_path,
            max_events=max_events,
        )
        rank_reports.append(result)
        adapter_paths[rank] = adapter_path

    selected = _select_rank(task, rank_reports)
    selected_rank = int(selected["rank"])
    selected_adapter = adapter_paths[selected_rank]
    model, adapter_payload = load_lora_task_model(
        base_checkpoint_path, selected_adapter, device=resolved_device
    )
    selected_validation_predictions = _predict(model, loaders["valid"], task=task, device=resolved_device)
    selected_validation_metrics, selected_validation_intervals = _metrics_and_intervals(
        task, selected_validation_predictions, bootstrap_samples=config.bootstrap_samples
    )
    test_predictions = _predict(model, loaders["test"], task=task, device=resolved_device)
    test_metrics, test_intervals = _metrics_and_intervals(
        task, test_predictions, bootstrap_samples=config.bootstrap_samples
    )

    report: dict[str, Any] = {
        "task": task,
        "base_checkpoint": str(Path(base_checkpoint_path).resolve()),
        "base_checkpoint_sha256": _file_sha256(Path(base_checkpoint_path)),
        "task_table": str(task_table_path),
        "processed_dir": str(processed_dir),
        "max_events": max_events,
        "device": str(resolved_device),
        "fine_tune_config": asdict(config),
        "adaptation": {
            "scope": "History Encoder QKV and feed-forward projections only",
            "selected_rank": selected_rank,
            "alpha": config.alpha,
            "adapter_checkpoint": str(selected_adapter),
            "trainable_parameter_count": int(adapter_payload["trainable_parameter_count"]),
            "backbone_parameter_count": int(adapter_payload["backbone_parameter_count"]),
            "trainable_fraction": float(adapter_payload["trainable_fraction"]),
        },
        "rank_validation_runs": rank_reports,
        "selected_validation_metrics": selected_validation_metrics,
        "selected_validation_account_cluster_bootstrap_95_intervals": selected_validation_intervals,
        "test_metrics": test_metrics,
        "test_account_cluster_bootstrap_95_intervals": test_intervals,
        "protocol": {
            "account_disjoint_splits": True,
            "cached_task_table_reused": True,
            "test_used_after_validation_rank_selection": True,
            "base_backbone_weights_frozen": True,
        },
    }
    if baseline_report_path is not None:
        report["tabular_baseline_report"] = _load_json(baseline_report_path)
    if frozen_probe_report_path is not None:
        report["frozen_embedding_probe_report"] = _load_json(frozen_probe_report_path)
    (output_dir / f"{task}_lora_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_lora_task_model(
    base_checkpoint_path: str | Path,
    adapter_checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[AccountTaskModel, dict[str, Any]]:
    """Recreate a frozen backbone and attach a portable saved LoRA adapter."""
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    payload = torch.load(adapter_checkpoint_path, map_location=resolved_device)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported LoRA adapter checkpoint format")
    backbone, _ = _load_backbone(base_checkpoint_path, device=resolved_device)
    lora_state = inject_history_lora(backbone, LoRAConfig(**payload["lora_config"]))
    model = AccountTaskModel(backbone, payload["task"]).to(resolved_device)
    adapter_tensors = dict(payload["adapter_state_dict"])
    adapter_tensors.update({f"head.{name}": value for name, value in payload["task_head_state_dict"].items()})
    missing, unexpected = model.load_state_dict(adapter_tensors, strict=False)
    if unexpected or any(name.startswith("head.") or name.endswith(("lora_a", "lora_b")) for name in missing):
        raise ValueError("LoRA adapter state is incompatible with the base backbone")
    if lora_state.trainable_parameter_count != int(payload["trainable_parameter_count"]):
        raise ValueError("LoRA adapter parameter count does not match its checkpoint metadata")
    model.eval()
    return model, payload


def _train_one_rank(
    *,
    task: TaskName,
    rank: int,
    base_checkpoint_path: str | Path,
    train_loader: DataLoader[dict[str, Any]],
    valid_loader: DataLoader[dict[str, Any]],
    output_dir: Path,
    config: FineTuneConfig,
    device: torch.device,
    task_table_path: Path,
    max_events: int,
) -> tuple[dict[str, Any], Path]:
    torch.manual_seed(config.seed + rank)
    backbone, checkpoint = _load_backbone(base_checkpoint_path, device=device)
    lora_state = inject_history_lora(
        backbone, LoRAConfig(rank=rank, alpha=config.alpha, dropout=config.lora_dropout)
    )
    model = AccountTaskModel(backbone, task).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": list(iter_lora_parameters(model.backbone)), "lr": config.lora_learning_rate},
            {"params": list(model.head.parameters()), "lr": config.head_learning_rate},
        ]
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_score = float("-inf") if task == "cashflow_stress" else float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    adapter_path = output_dir / f"{task}_lora_rank_{rank:02d}.pt"

    for epoch in range(1, config.max_epochs + 1):
        started_at = perf_counter()
        model.train()
        loss_sum = 0.0
        example_count = 0
        for raw_batch in train_loader:
            batch = _move_tensors(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model.forward_from_batch(batch)
                loss = _loss(task, predictions, batch["targets"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(predictions.shape[0])
            loss_sum += float(loss.detach()) * batch_size
            example_count += batch_size

        validation = _predict(model, valid_loader, task=task, device=device)
        validation_metrics, _ = _metrics_and_intervals(task, validation, bootstrap_samples=0)
        selection_score = _selection_score(task, validation_metrics)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss_sum / max(example_count, 1),
                "validation_selection_score": selection_score,
                "seconds": perf_counter() - started_at,
            }
        )
        if _is_improved(task, selection_score, best_score):
            best_score = selection_score
            best_epoch = epoch
            bad_epochs = 0
            _save_adapter_checkpoint(
                adapter_path,
                model=model,
                task=task,
                base_checkpoint_path=base_checkpoint_path,
                task_table_path=task_table_path,
                lora_state=lora_state,
                max_events=max_events,
                config=config,
                validation_metrics=validation_metrics,
                epoch=epoch,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    best_model, _ = load_lora_task_model(base_checkpoint_path, adapter_path, device=device)
    best_validation = _predict(best_model, valid_loader, task=task, device=device)
    validation_metrics, _ = _metrics_and_intervals(task, best_validation, bootstrap_samples=0)
    return (
        {
            "rank": rank,
            "best_epoch": best_epoch,
            "validation_metrics": validation_metrics,
            "training_history": history,
            "adapter_checkpoint": str(adapter_path),
        },
        adapter_path,
    )


def _build_task_datasets(
    task_table: pd.DataFrame, *, processed_dir: Path, base_checkpoint_path: str | Path
) -> tuple[dict[str, FinBERTLiteCzechDataset], int]:
    _, checkpoint = _load_backbone(base_checkpoint_path, device=torch.device("cpu"))
    tokenizers = load_tokenizer_bundle(Path(base_checkpoint_path).resolve().parent / "tokenizers")
    max_events = int(checkpoint.get("max_events", 64))
    lifelong_events_path = processed_dir / "lifelong_events.parquet"
    if not lifelong_events_path.exists():
        raise FileNotFoundError(f"processed lifelong events not found: {lifelong_events_path}")
    datasets: dict[str, FinBERTLiteCzechDataset] = {}
    for split in ("train", "valid", "test"):
        datasets[split] = FinBERTLiteCzechDataset(
            processed_dir / f"events_{split}.parquet",
            processed_dir / f"profile_{split}.parquet",
            lifelong_events_path,
            tokenizers,
            max_events=max_events,
            random_cutoff=False,
            task_table=task_table.loc[task_table["split"].eq(split)],
        )
    return datasets, max_events


def _build_loaders(
    datasets: dict[str, FinBERTLiteCzechDataset], *, config: FineTuneConfig, device: torch.device
) -> dict[str, DataLoader[dict[str, Any]]]:
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers if device.type == "cuda" else 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_account_records,
    }
    return {
        "train": DataLoader(
            datasets["train"], shuffle=True,
            generator=torch.Generator().manual_seed(config.seed), **common
        ),
        "valid": DataLoader(datasets["valid"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def _load_backbone(path: str | Path, *, device: torch.device) -> tuple[EventMLMDemoModel, dict[str, Any]]:
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"base checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    required = {"model_state_dict", "model_config", "vocabulary_size"}
    if missing := required.difference(checkpoint):
        raise ValueError(f"base checkpoint is missing fields: {sorted(missing)}")
    model = EventMLMDemoModel(
        int(checkpoint["vocabulary_size"]), TransformerConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def _predict(
    model: AccountTaskModel,
    loader: DataLoader[dict[str, Any]],
    *,
    task: TaskName,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    account_ids: list[np.ndarray] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_tensors(raw_batch, device)
            logits_or_values = model.forward_from_batch(batch)
            values = torch.sigmoid(logits_or_values) if task == "cashflow_stress" else logits_or_values
            targets.append(raw_batch["targets"].cpu().numpy())
            predictions.append(values.cpu().numpy())
            account_ids.append(raw_batch["account_ids"].cpu().numpy())
    return {
        "targets": np.concatenate(targets),
        "predictions": np.concatenate(predictions),
        "account_ids": np.concatenate(account_ids),
    }


def _metrics_and_intervals(
    task: TaskName, predictions: dict[str, np.ndarray], *, bootstrap_samples: int = 1_000
) -> tuple[dict[str, float | int], dict[str, tuple[float, float]]]:
    if task == "cashflow_stress":
        metrics = binary_classification_metrics(predictions["targets"], predictions["predictions"])
        intervals = (
            clustered_binary_bootstrap_intervals(
                predictions["targets"], predictions["predictions"], predictions["account_ids"], samples=bootstrap_samples
            )
            if bootstrap_samples
            else {}
        )
    else:
        metrics = regression_metrics(predictions["targets"], predictions["predictions"])
        intervals = (
            clustered_regression_bootstrap_intervals(
                predictions["targets"], predictions["predictions"], predictions["account_ids"], samples=bootstrap_samples
            )
            if bootstrap_samples
            else {}
        )
    return metrics, intervals


def _loss(task: TaskName, predictions: Tensor, targets: Tensor) -> Tensor:
    targets = targets.to(device=predictions.device, dtype=predictions.dtype)
    if task == "cashflow_stress":
        return F.binary_cross_entropy_with_logits(predictions, targets)
    return F.smooth_l1_loss(predictions, targets)


def _selection_score(task: TaskName, metrics: dict[str, float | int]) -> float:
    return float(metrics["average_precision"] if task == "cashflow_stress" else metrics["mae"])


def _is_improved(task: TaskName, candidate: float, incumbent: float) -> bool:
    return candidate > incumbent if task == "cashflow_stress" else candidate < incumbent


def _select_rank(task: TaskName, reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one LoRA rank run is required")
    key = lambda report: _selection_score(task, report["validation_metrics"])
    return max(reports, key=key) if task == "cashflow_stress" else min(reports, key=key)


def _save_adapter_checkpoint(
    path: Path,
    *,
    model: AccountTaskModel,
    task: TaskName,
    base_checkpoint_path: str | Path,
    task_table_path: Path,
    lora_state: Any,
    max_events: int,
    config: FineTuneConfig,
    validation_metrics: dict[str, float | int],
    epoch: int,
) -> None:
    payload = {
        "format_version": 1,
        "task": task,
        "base_checkpoint": str(Path(base_checkpoint_path).resolve()),
        "base_checkpoint_sha256": _file_sha256(Path(base_checkpoint_path)),
        "task_table": str(task_table_path),
        "max_events": max_events,
        "lora_config": asdict(lora_state.config),
        "target_module_names": lora_state.target_module_names,
        "trainable_parameter_count": lora_state.trainable_parameter_count,
        "backbone_parameter_count": lora_state.backbone_parameter_count,
        "trainable_fraction": lora_state.trainable_fraction,
        "fine_tune_config": asdict(config),
        "best_epoch": epoch,
        "validation_metrics": validation_metrics,
        "adapter_state_dict": adapter_state_dict(model),
        "task_head_state_dict": {name: value.detach().cpu() for name, value in model.head.state_dict().items()},
    }
    torch.save(payload, path)


def _validate_task_table(task_table: pd.DataFrame, task: TaskName) -> None:
    required = {"sample_id", "account_id", "cutoff_time", "split", "target"}
    if missing := required.difference(task_table.columns):
        raise ValueError(f"task table is missing columns: {sorted(missing)}")
    if task_table["sample_id"].duplicated().any():
        raise ValueError("task table sample IDs must be unique")
    if not task_table["sample_id"].astype(str).str.startswith(f"{task}:").all():
        raise ValueError("task table does not match requested task")
    if set(task_table["split"].unique()) != {"train", "valid", "test"}:
        raise ValueError("task table must contain exactly train, valid, and test splits")


def _move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
