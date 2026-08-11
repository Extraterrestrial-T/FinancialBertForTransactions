"""Run a real-data end-to-end training pass for the PRAGMA-lite model.

The default is deliberately small: four train batches and one validation
batch.  It nevertheless exercises the complete path:

    parquet -> Dataset -> collation -> MLM corruption -> Event/Profile/
    History encoders -> MLM loss -> backward -> AdamW update.

From the repository root:

    python scripts/run_training_smoke.py

To make a CPU-only full epoch run (slow), use ``--steps 0``.  This is still a
training smoke script: it intentionally does not add checkpointing, a learning
rate schedule, or epoch-level validation aggregation yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pragma_lite import (  # noqa: E402
    EventMLMDemoModel,
    FinBERTLiteCzechDataset,
    TransformerConfig,
    apply_value_mlm_mask,
    collate_account_records,
    fit_tokenizer_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="number of train batches; use 0 for the entire train loader",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device, for example 'cpu' or 'cuda'; defaults to CUDA when available",
    )
    args = parser.parse_args()
    if args.steps < 0 or args.batch_size < 1 or args.max_events < 1:
        parser.error("steps must be non-negative; batch-size and max-events must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    return args


def move_tensors_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    """Keep account metadata on CPU while moving model tensors to ``device``."""
    return {
        name: value.to(device) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def model_forward(model: EventMLMDemoModel, batch: dict[str, object]):
    """Call the model in the same tensor order emitted by the data handler."""
    return model(
        batch["event_key_ids"],  # type: ignore[arg-type]
        batch["event_value_ids"],  # type: ignore[arg-type]
        batch["event_mask"],  # type: ignore[arg-type]
        batch["profile_key_ids"],  # type: ignore[arg-type]
        batch["profile_value_ids"],  # type: ignore[arg-type]
        batch["profile_mask"],  # type: ignore[arg-type]
        batch["profile_rope_time"],  # type: ignore[arg-type]
        batch["history_rope_time"],  # type: ignore[arg-type]
        batch["calendar_features"],  # type: ignore[arg-type]
    )


def masked_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Cross-entropy over only selected MLM targets (``-100`` is ignored)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    processed = ROOT / "data" / "processed" / "czech_bank"
    lifelong_events_path = processed / "lifelong_events.parquet"
    paths = {
        "train_events": processed / "events_train.parquet",
        "train_profiles": processed / "profile_train.parquet",
        "valid_events": processed / "events_valid.parquet",
        "valid_profiles": processed / "profile_valid.parquet",
    }
    missing = [str(path) for path in (*paths.values(), lifelong_events_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required Czech data:\n" + "\n".join(missing))

    # Numeric bucket edges and profile/lifelong vocabularies are fitted once,
    # strictly on train data.  Validation reuses the resulting bundle.
    train_events = pd.read_parquet(paths["train_events"])
    train_profiles = pd.read_parquet(paths["train_profiles"])
    tokenizers = fit_tokenizer_bundle(train_events, train_profiles, lifelong_events_path)
    del train_events, train_profiles

    train_dataset = FinBERTLiteCzechDataset(
        paths["train_events"],
        paths["train_profiles"],
        lifelong_events_path,
        tokenizers,
        max_events=args.max_events,
        random_cutoff=True,
        seed=17,
    )
    valid_dataset = FinBERTLiteCzechDataset(
        paths["valid_events"],
        paths["valid_profiles"],
        lifelong_events_path,
        tokenizers,
        max_events=args.max_events,
        random_cutoff=False,
        seed=17,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_account_records,
        generator=torch.Generator().manual_seed(23),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_account_records,
    )

    config = TransformerConfig(d_model=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.1)
    model = EventMLMDemoModel(len(tokenizers.event.token_to_id), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    train_steps = len(train_loader) if args.steps == 0 else min(args.steps, len(train_loader))
    if train_steps == 0:
        raise RuntimeError("the training dataset is empty")

    print(f"device: {device}")
    print(f"accounts: train={len(train_dataset):,}, valid={len(valid_dataset):,}")
    print(f"model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"running {train_steps} training step(s), batch_size={args.batch_size}, max_events={args.max_events}")

    model.train()
    masking_generator = torch.Generator().manual_seed(29)
    for step, unmasked_batch in enumerate(train_loader, start=1):
        if step > train_steps:
            break
        batch = move_tensors_to_device(unmasked_batch, device)
        batch = apply_value_mlm_mask(batch, generator=masking_generator)
        optimizer.zero_grad(set_to_none=True)
        output = model_forward(model, batch)
        loss = masked_loss(output.logits, batch["mlm_labels"])  # type: ignore[arg-type]
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        targets = int(batch["mlm_mask"].sum().item())  # type: ignore[union-attr]
        print(
            f"step {step:>3}/{train_steps}: loss={loss.item():.4f} "
            f"targets={targets:>4} grad_norm={float(gradient_norm):.3f}"
        )

    # A deterministic latest-cutoff validation sample confirms that evaluation
    # uses the train-fitted tokenizers but does not update model weights.
    model.eval()
    with torch.no_grad():
        validation_batch = move_tensors_to_device(next(iter(valid_loader)), device)
        validation_batch = apply_value_mlm_mask(
            validation_batch,
            generator=torch.Generator().manual_seed(31),
        )
        validation_output = model_forward(model, validation_batch)
        validation_loss = masked_loss(
            validation_output.logits,
            validation_batch["mlm_labels"],  # type: ignore[arg-type]
        )
    print(
        f"validation batch: loss={validation_loss.item():.4f} "
        f"targets={int(validation_batch['mlm_mask'].sum().item())} "  # type: ignore[union-attr]
        f"account_embeddings={tuple(validation_output.account_embedding.shape)}"
    )


if __name__ == "__main__":
    main()
