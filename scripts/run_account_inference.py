"""Run one cutoff-safe Czech account snapshot through a base model plus adapter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pragma_lite import AccountTaskPredictor, load_processed_czech_account_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "czech_bank",
        help="Directory containing the committed Czech processed artifacts.",
    )
    parser.add_argument("--account-id", type=int, required=True)
    parser.add_argument(
        "--cutoff",
        type=datetime.fromisoformat,
        default=None,
        help="Optional ISO timestamp. Defaults to the account's last known transaction.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = AccountTaskPredictor.from_checkpoints(
        args.base_checkpoint,
        args.adapter,
        device=args.device,
    )
    prepared = load_processed_czech_account_sample(
        args.processed_dir,
        args.account_id,
        cutoff_time=args.cutoff,
    )
    print(json.dumps(asdict(predictor.predict(prepared)), indent=2, default=str))


if __name__ == "__main__":
    main()
