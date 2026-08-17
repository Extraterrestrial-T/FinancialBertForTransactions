"""Discovery helpers for a portable PRAGMA-lite model release.

The web explorer and any future deployment should not hard-code an experiment
directory.  Instead they recognise a base checkpoint by the tokenizer state
that must travel with it, and recognise adapters from their saved task
metadata.  An adapter path returned here is safe to hand to
``AccountTaskPredictor.from_checkpoints``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


_REQUIRED_TOKENIZERS = frozenset(
    {
        "event_tokenizer_state.json",
        "profile_tokenizer_state.json",
        "lifelong_tokenizer_state.json",
    }
)


@dataclass(frozen=True, slots=True)
class AdapterArtifact:
    """A task adapter found in a curated artifact release."""

    identifier: str
    path: Path
    task: str
    label: str
    rank: int | None


def discover_base_checkpoint(artifacts_dir: str | Path) -> Path | None:
    """Find the release's inference checkpoint and its tokenizer contract."""
    for checkpoint in sorted(Path(artifacts_dir).rglob("best.pt")):
        tokenizer_dir = checkpoint.parent / "tokenizers"
        if not tokenizer_dir.is_dir():
            continue
        names = {path.name for path in tokenizer_dir.iterdir() if path.is_file()}
        if _REQUIRED_TOKENIZERS.issubset(names):
            return checkpoint.resolve()
    return None


def discover_task_adapters(artifacts_dir: str | Path) -> tuple[AdapterArtifact, ...]:
    """Return adapter files carrying the expected saved task metadata.

    ``torch.load`` is appropriate here only because this scans the repository's
    own curated artifact directory.  Do not point it at untrusted uploads.
    """
    root = Path(artifacts_dir).resolve()
    found: list[AdapterArtifact] = []
    for path in sorted(root.rglob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            task = str(payload["task"])
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            continue
        if task not in {"cashflow_stress", "future_value"}:
            continue
        rank = _rank_from_payload_or_name(payload, path.name)
        task_label = {
            "cashflow_stress": "60-day cash-flow stress",
            "future_value": "180-day transaction-volume proxy",
        }[task]
        rank_label = f" · rank {rank}" if rank is not None else ""
        found.append(
            AdapterArtifact(
                identifier=path.relative_to(root).as_posix(),
                path=path.resolve(),
                task=task,
                label=f"{task_label}{rank_label}",
                rank=rank,
            )
        )
    return tuple(found)


def _rank_from_payload_or_name(payload: object, filename: str) -> int | None:
    if isinstance(payload, dict):
        config = payload.get("lora_config")
        if isinstance(config, dict) and isinstance(config.get("rank"), int):
            return int(config["rank"])
    marker = "rank_"
    if marker in filename:
        suffix = filename.split(marker, maxsplit=1)[1].split(".", maxsplit=1)[0]
        if suffix.isdigit():
            return int(suffix)
    return None
