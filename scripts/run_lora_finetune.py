"""Fine-tune a frozen PRAGMA-lite checkpoint with History Encoder LoRA.

The task table must already exist.  The recommended notebook sequence is:
``03_frozen_downstream_evaluation.ipynb`` to create and inspect the cached
table, then either LoRA notebook to adapt on exactly those dated examples.

Example::

    python scripts/run_lora_finetune.py --task future_value \
      --checkpoint /content/drive/MyDrive/FinancialBertForTransactions/checkpoints/pragma_lite_mlm/best.pt \
      --task-table /content/drive/MyDrive/FinancialBertForTransactions/reports/future_value_task_table.parquet \
      --output-dir /content/drive/MyDrive/FinancialBertForTransactions/adapters/future_value
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pragma_lite.training import FineTuneConfig, run_lora_finetuning  # noqa: E402


def _parse_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from error
    if not ranks or any(rank < 1 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must contain positive integers")
    return ranks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("cashflow_stress", "future_value"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="frozen MLM base checkpoint")
    parser.add_argument("--task-table", type=Path, required=True, help="cached, cutoff-safe task table")
    parser.add_argument("--output-dir", type=Path, required=True, help="adapter and JSON report directory")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed" / "czech_bank")
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--frozen-probe-report", type=Path, default=None)
    parser.add_argument("--ranks", type=_parse_ranks, default=(4, 8, 16))
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FineTuneConfig(
        ranks=args.ranks,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        bootstrap_samples=args.bootstrap_samples,
    )
    report = run_lora_finetuning(
        task=args.task,
        base_checkpoint_path=args.checkpoint,
        task_table_path=args.task_table,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
        baseline_report_path=args.baseline_report,
        frozen_probe_report_path=args.frozen_probe_report,
    )
    metrics = report["test_metrics"]
    criterion = "average_precision" if args.task == "cashflow_stress" else "mae"
    print(
        f"selected rank {report['adaptation']['selected_rank']}; "
        f"test {criterion}={metrics[criterion]:.4f}; "
        f"report={args.output_dir / f'{args.task}_lora_report.json'}"
    )


if __name__ == "__main__":
    main()
