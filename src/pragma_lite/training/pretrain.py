"""Reusable masked-financial-event pre-training loop for PRAGMA-lite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data import (
    FinBERTLiteCzechDataset,
    apply_value_mlm_mask,
    collate_account_records,
    fit_tokenizer_bundle,
    save_tokenizer_bundle,
)
from ..models import EventMLMDemoModel, TransformerConfig


@dataclass(frozen=True, slots=True)
class PretrainingConfig:
    """Settings for a compact bidirectional value-MLM training run."""

    model: TransformerConfig = TransformerConfig(
        d_model=128, num_heads=8, num_layers=2, ffn_dim=256, dropout=0.1
    )
    max_events: int = 256
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    num_workers: int = 2
    seed: int = 17
    mask_probability: float = 0.15

    def __post_init__(self) -> None:
        if self.max_events < 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("max_events, batch_size, and epochs must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 < self.mask_probability <= 1.0:
            raise ValueError("mask_probability must be in (0, 1]")


def run_pretraining(
    *,
    processed_dir: str | Path,
    checkpoint_dir: str | Path,
    config: PretrainingConfig | None = None,
    device: str | torch.device | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Fit tokenizers on train only, train MLM, and persist the best checkpoint.

    The processed directory is read-only.  Only the supplied checkpoint
    directory is written, which lets Colab use its local SSD for data and
    Google Drive solely for durable experiment artifacts.
    """
    config = PretrainingConfig() if config is None else config
    processed_dir = Path(processed_dir).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = _required_processed_paths(processed_dir)
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(config.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    last_checkpoint = checkpoint_dir / "last.pt"
    if resume:
        if not last_checkpoint.exists():
            raise FileNotFoundError(f"cannot resume: checkpoint not found: {last_checkpoint}")
        from ..data import load_tokenizer_bundle

        tokenizers = load_tokenizer_bundle(checkpoint_dir / "tokenizers")
        resume_payload = torch.load(last_checkpoint, map_location=resolved_device)
        _validate_resume_checkpoint(resume_payload, config)
    else:
        train_events = pd.read_parquet(paths["train_events"])
        train_profiles = pd.read_parquet(paths["train_profiles"])
        tokenizers = fit_tokenizer_bundle(train_events, train_profiles, paths["lifelong_events"])
        save_tokenizer_bundle(tokenizers, checkpoint_dir / "tokenizers")
        resume_payload = None
    vocabulary_size = len(tokenizers.event.token_to_id)
    if not resume:
        del train_events, train_profiles

    datasets = {
        "train": FinBERTLiteCzechDataset(
            paths["train_events"], paths["train_profiles"], paths["lifelong_events"], tokenizers,
            max_events=config.max_events, random_cutoff=True, seed=config.seed,
        ),
        "valid": FinBERTLiteCzechDataset(
            paths["valid_events"], paths["valid_profiles"], paths["lifelong_events"], tokenizers,
            max_events=config.max_events, random_cutoff=False, seed=config.seed,
        ),
    }
    loaders = _build_pretraining_loaders(datasets, config=config, device=resolved_device)
    model = EventMLMDemoModel(vocabulary_size, config.model).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    amp_enabled = resolved_device.type == "cuda"
    scaler = torch.amp.GradScaler(resolved_device.type, enabled=amp_enabled)

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        best_validation_loss = float(resume_payload["best_validation_loss"])
        history = list(resume_payload["history"])
        first_epoch = int(resume_payload["epoch"]) + 1
    else:
        best_validation_loss = float("inf")
        history: list[dict[str, float]] = []
        first_epoch = 1
    best_checkpoint = checkpoint_dir / "best.pt"
    for epoch in range(first_epoch, config.epochs + 1):
        started_at = perf_counter()
        datasets["train"].set_epoch(epoch)
        train_loss = _run_mlm_epoch(
            model, loaders["train"], optimizer=optimizer, scaler=scaler, config=config,
            device=resolved_device, masking_seed=config.seed + epoch,
        )
        validation_loss = _evaluate_mlm(
            model, loaders["valid"], config=config, device=resolved_device,
            masking_seed=config.seed + 10_000 + epoch,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
            "seconds": perf_counter() - started_at,
        }
        history.append(row)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            _save_pretraining_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                vocabulary_size=vocabulary_size,
                epoch=epoch,
                validation_loss=validation_loss,
                scheduler=scheduler,
                best_validation_loss=best_validation_loss,
                history=history,
            )
        scheduler.step()
        _save_pretraining_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            vocabulary_size=vocabulary_size,
            epoch=epoch,
            validation_loss=validation_loss,
            best_validation_loss=best_validation_loss,
            history=history,
        )

    report: dict[str, Any] = {
        "checkpoint": str(best_checkpoint),
        "device": str(resolved_device),
        "vocabulary_size": vocabulary_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "pretraining_config": _serialise_pretraining_config(config),
        "best_validation_loss": best_validation_loss,
        "resumed": resume,
        "history": history,
        "data_protocol": {
            "processed_data_read_only": True,
            "numeric_and_categorical_tokenizer_fit_split": "train",
            "validation_uses_latest_account_cutoff": True,
            "training_uses_per_epoch_random_account_cutoffs": True,
        },
    }
    (checkpoint_dir / "pretrain_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _required_processed_paths(processed_dir: Path) -> dict[str, Path]:
    paths = {
        "train_events": processed_dir / "events_train.parquet",
        "train_profiles": processed_dir / "profile_train.parquet",
        "valid_events": processed_dir / "events_valid.parquet",
        "valid_profiles": processed_dir / "profile_valid.parquet",
        "lifelong_events": processed_dir / "lifelong_events.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required processed Czech artifacts:\n" + "\n".join(missing))
    return paths


def _build_pretraining_loaders(
    datasets: dict[str, FinBERTLiteCzechDataset], *, config: PretrainingConfig, device: torch.device
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
    }


def _run_mlm_epoch(
    model: EventMLMDemoModel,
    loader: DataLoader[dict[str, Any]],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: PretrainingConfig,
    device: torch.device,
    masking_seed: int,
) -> float:
    model.train()
    generator = torch.Generator(device=device.type).manual_seed(masking_seed)
    loss_sum = 0.0
    target_count = 0
    for raw_batch in loader:
        batch = apply_value_mlm_mask(
            _move_tensors(raw_batch, device),
            token_mask_probability=config.mask_probability,
            generator=generator,
        )
        labels = batch["mlm_labels"]
        selected = int(labels.ne(-100).sum().item())
        if selected == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = _forward(model, batch).logits
            loss = _masked_cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach()) * selected
        target_count += selected
    if target_count == 0:
        raise RuntimeError("MLM masking selected no targets in this training epoch")
    return loss_sum / target_count


def _evaluate_mlm(
    model: EventMLMDemoModel,
    loader: DataLoader[dict[str, Any]],
    *,
    config: PretrainingConfig,
    device: torch.device,
    masking_seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=device.type).manual_seed(masking_seed)
    loss_sum = 0.0
    target_count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = apply_value_mlm_mask(
                _move_tensors(raw_batch, device),
                token_mask_probability=config.mask_probability,
                generator=generator,
            )
            labels = batch["mlm_labels"]
            selected = int(labels.ne(-100).sum().item())
            if selected == 0:
                continue
            logits = _forward(model, batch).logits
            loss_sum += float(_masked_cross_entropy(logits, labels)) * selected
            target_count += selected
    if target_count == 0:
        raise RuntimeError("MLM masking selected no targets in validation")
    return loss_sum / target_count


def _forward(model: EventMLMDemoModel, batch: dict[str, Any]):
    return model(
        batch["event_key_ids"], batch["event_value_ids"], batch["event_mask"],
        batch["profile_key_ids"], batch["profile_value_ids"], batch["profile_mask"],
        batch["profile_rope_time"], batch["history_rope_time"], batch["calendar_features"],
    )


def _masked_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def _move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for name, value in batch.items()}


def _save_pretraining_checkpoint(
    path: Path,
    *,
    model: EventMLMDemoModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: PretrainingConfig,
    vocabulary_size: int,
    epoch: int,
    validation_loss: float,
    best_validation_loss: float,
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": asdict(config.model),
            "pretraining_config": _serialise_pretraining_config(config),
            "vocabulary_size": vocabulary_size,
            "max_events": config.max_events,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
            "history": history,
        },
        path,
    )


def _serialise_pretraining_config(config: PretrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["model"] = asdict(config.model)
    return payload


def _validate_resume_checkpoint(checkpoint: dict[str, Any], config: PretrainingConfig) -> None:
    required = {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "model_config",
        "pretraining_config", "vocabulary_size", "max_events", "epoch", "best_validation_loss", "history",
    }
    if missing := required.difference(checkpoint):
        raise ValueError(f"resume checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["model_config"] != asdict(config.model):
        raise ValueError("resume checkpoint model configuration does not match this run")
    if int(checkpoint["max_events"]) != config.max_events:
        raise ValueError("resume checkpoint max_events does not match this run")
    if checkpoint["pretraining_config"] != _serialise_pretraining_config(config):
        raise ValueError(
            "resume checkpoint training configuration does not match this run; "
            "keep the original total epoch count so the saved scheduler remains valid"
        )
