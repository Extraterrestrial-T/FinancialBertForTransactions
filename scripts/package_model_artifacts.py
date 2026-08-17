"""Create the small, commit-ready PRAGMA-lite model release directory.

Example::

    python scripts/package_model_artifacts.py \
      --base-checkpoint C:/models/best.pt \
      --future-adapter C:/models/future_value_lora_rank_04.pt \
      --cashflow-adapter C:/models/cashflow_stress_lora_rank_16.pt

The command refuses to overwrite a release and verifies that each adapter was
trained against the supplied base checkpoint before copying any artifact.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import shutil
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--future-adapter", type=Path, required=True)
    parser.add_argument("--cashflow-adapter", type=Path, required=True)
    parser.add_argument("--future-report", type=Path, default=None)
    parser.add_argument("--cashflow-report", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "app_assets" / "artifacts"
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_adapter(adapter: Path, *, expected_task: str, base_hash: str) -> None:
    payload = torch.load(adapter, map_location="cpu", weights_only=False)
    if payload.get("task") != expected_task:
        raise ValueError(f"{adapter} is not a {expected_task} adapter")
    if payload.get("base_checkpoint_sha256") != base_hash:
        raise ValueError(f"{adapter} was trained against a different base checkpoint")


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    base = args.base_checkpoint.resolve()
    future_adapter = args.future_adapter.resolve()
    cashflow_adapter = args.cashflow_adapter.resolve()
    output = args.output_dir.resolve()
    tokenizer_source = base.parent / "tokenizers"
    tokenizer_names = (
        "event_tokenizer_state.json",
        "profile_tokenizer_state.json",
        "lifelong_tokenizer_state.json",
    )
    required_sources = (base, future_adapter, cashflow_adapter) + tuple(
        tokenizer_source / name for name in tokenizer_names
    )
    for source in required_sources:
        if not source.is_file():
            raise FileNotFoundError(f"required release artifact is missing: {source}")
    for report in (args.future_report, args.cashflow_report):
        if report is not None and not report.is_file():
            raise FileNotFoundError(f"optional release report is missing: {report}")

    base_hash = file_sha256(base)
    validate_adapter(future_adapter, expected_task="future_value", base_hash=base_hash)
    validate_adapter(cashflow_adapter, expected_task="cashflow_stress", base_hash=base_hash)

    destinations = (
        output / "models" / "checkpoints_swoll" / "pragma_lite_mlm" / "best.pt",
        output / "adaptors" / "future_value" / "future_value_lora_rank_04.pt",
        output / "adaptors" / "cashflow_stress" / "cashflow_stress_lora_rank_16.pt",
    )
    existing = [path for path in destinations if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"release outputs already exist; refusing to overwrite: {names}")

    base_destination = output / "models" / "checkpoints_swoll" / "pragma_lite_mlm"
    future_adapter_destination = output / "adaptors" / "future_value"
    cashflow_adapter_destination = output / "adaptors" / "cashflow_stress"
    reports_destination = output / "reports"
    tokenizer_destination = base_destination / "tokenizers"
    tokenizer_destination.mkdir(parents=True, exist_ok=False)
    future_adapter_destination.mkdir(parents=True, exist_ok=False)
    cashflow_adapter_destination.mkdir(parents=True, exist_ok=False)
    copy_required(base, base_destination / "best.pt")
    for name in tokenizer_names:
        copy_required(tokenizer_source / name, tokenizer_destination / name)
    copy_required(future_adapter, future_adapter_destination / "future_value_lora_rank_04.pt")
    copy_required(cashflow_adapter, cashflow_adapter_destination / "cashflow_stress_lora_rank_16.pt")

    optional_reports = (
        (args.future_report, reports_destination / "future_value_lora_report.json"),
        (args.cashflow_report, reports_destination / "cashflow_stress_lora_report.json"),
    )
    for source, destination in optional_reports:
        if source is not None:
            reports_destination.mkdir(parents=True, exist_ok=True)
            copy_required(source.resolve(), destination)

    print(f"created verified model release at {output}")
    print(f"base SHA-256: {base_hash}")


if __name__ == "__main__":
    main()
