"""Minimal LoRA support for PRAGMA-lite History Encoder adaptation.

Only the History Encoder Q/K/V and feed-forward projections are adapted. The
rest of the MLM backbone stays frozen, which makes the trainable footprint
small and keeps a task adapter portable beside its base checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..models import EventMLMDemoModel


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """Low-rank adapter settings shared across selected linear projections."""

    rank: int = 8
    alpha: float = 8.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")


class LoRALinear(nn.Module):
    """Frozen linear map plus a trainable low-rank residual.

    ``lora_b`` starts at zero, so replacing an ``nn.Linear`` with this module
    leaves the model output exactly unchanged before its first optimisation
    step. The residual is ``(alpha / rank) * B @ A @ x``.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can wrap only nn.Linear modules")
        self.base = base
        self.config = config
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.rank))
        self.dropout = nn.Dropout(config.dropout)
        self.scaling = config.alpha / config.rank
        nn.init.kaiming_uniform_(self.lora_a, a=sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        residual = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return self.base(x) + residual * self.scaling


@dataclass(frozen=True, slots=True)
class HistoryLoRAState:
    """Names and parameter counts emitted after History Encoder injection."""

    config: LoRAConfig
    target_module_names: tuple[str, ...]
    trainable_parameter_count: int
    backbone_parameter_count: int

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_parameter_count / self.backbone_parameter_count


def inject_history_lora(model: EventMLMDemoModel, config: LoRAConfig) -> HistoryLoRAState:
    """Freeze a backbone and replace only History Encoder QKV/MLP linears.

    This mutates a freshly loaded model in place. Calling it twice is an
    error, preventing accidental adapter stacking.
    """
    # Capture this before adding adapter matrices: it describes the actual
    # pretrained backbone rather than the backbone-plus-adapter module tree.
    backbone_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    target_names: list[str] = []
    for block_index, block in enumerate(model.history_encoder.blocks):
        modules = {
            "attention.q_proj": block.attention.q_proj,
            "attention.k_proj": block.attention.k_proj,
            "attention.v_proj": block.attention.v_proj,
            "feed_forward.input_proj": block.feed_forward.input_proj,
            "feed_forward.output_proj": block.feed_forward.output_proj,
        }
        for relative_name, module in modules.items():
            if isinstance(module, LoRALinear):
                raise ValueError("History Encoder already has LoRA adapters")
            if not isinstance(module, nn.Linear):
                raise TypeError(f"expected nn.Linear at {relative_name}")
            parent, attribute = _parent_and_attribute(block, relative_name)
            setattr(parent, attribute, LoRALinear(module, config))
            target_names.append(f"history_encoder.blocks.{block_index}.{relative_name}")

    trainable = tuple(iter_lora_parameters(model))
    return HistoryLoRAState(
        config=config,
        target_module_names=tuple(target_names),
        trainable_parameter_count=sum(parameter.numel() for parameter in trainable),
        backbone_parameter_count=backbone_parameter_count,
    )


def iter_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield exactly the trainable adapter tensors, never frozen base weights."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b


def adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Extract portable LoRA tensors without duplicating frozen backbone weights."""
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.endswith("lora_a") or name.endswith("lora_b")
    }


def _parent_and_attribute(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
