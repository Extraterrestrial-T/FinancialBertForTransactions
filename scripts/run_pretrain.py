"""Run PRAGMA-lite value-MLM pre-training from processed Czech artifacts.

The notebook supplies an output directory on Google Drive for durable
checkpoints.  The repository itself and processed data should remain on
Colab's local ``/content`` SSD while this command runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pragma_lite.training import PretrainingConfig, run_pretraining  # noqa: E402
from pragma_lite.models import TransformerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed" / "czech_bank")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="resume checkpoint-dir/last.pt at an epoch boundary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PretrainingConfig(
        model=TransformerConfig(
            d_model=args.d_model,
            num_heads=args.heads,
            num_layers=args.layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
        ),
        max_events=args.max_events,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    report = run_pretraining(
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        config=config,
        device=args.device,
        resume=args.resume,
    )
    print(
        f"finished {args.epochs} epochs; best validation MLM loss="
        f"{report['best_validation_loss']:.4f}; checkpoint={report['checkpoint']}"
    )


if __name__ == "__main__":
    main()
