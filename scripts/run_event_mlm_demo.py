"""Run a CPU-only synthetic forward/backward pass through the Event MLM model.

From the repository root:
    python scripts/run_event_mlm_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pragma_lite import (  # noqa: E402
    EventMLMDemoModel,
    EventTokenizer,
    TransformerConfig,
    apply_value_mlm_mask,
)


def main() -> None:
    torch.manual_seed(7)
    vocabulary_size = len(EventTokenizer().token_to_id)
    batch_size, event_count, field_count = 2, 4, 6
    event_key_ids = torch.tensor(
        [4, 5, 6, 7, 8, 9], dtype=torch.long
    ).view(1, 1, field_count).expand(batch_size, event_count, -1).clone()
    event_value_ids = torch.randint(
        low=4,
        high=vocabulary_size,
        size=(batch_size, event_count, field_count),
        dtype=torch.long,
    )
    event_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )
    event_key_ids[~event_mask] = 0
    event_value_ids[~event_mask] = 0

    profile_key_ids = torch.tensor([[71, 72, 73], [71, 72, 0]], dtype=torch.long)
    profile_value_ids = torch.randint(4, vocabulary_size, (batch_size, 3), dtype=torch.long)
    profile_value_ids[1, 2] = 0
    profile_mask = torch.tensor([[True, True, True], [True, True, False]])
    profile_rope_time = torch.tensor([[0.0, 6.0, 12.0], [0.0, 4.0, 0.0]])

    calendar_features = torch.randn(batch_size, event_count, 4)
    calendar_features[~event_mask] = 0.0
    history_rope_time = torch.tensor([[12.0, 8.0, 4.0, 0.0], [5.0, 0.0, 0.0, 0.0]])

    masked = apply_value_mlm_mask(
        {"event_value_ids": event_value_ids, "event_mask": event_mask},
        token_mask_probability=1.0,
        event_mask_probability=0.0,
        field_mask_probability=0.0,
        unk_replacement_probability=0.0,
        generator=torch.Generator().manual_seed(11),
    )
    model = EventMLMDemoModel(
        vocabulary_size,
        TransformerConfig(d_model=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.0),
    )
    output = model(
        event_key_ids,
        masked["event_value_ids"],
        event_mask,
        profile_key_ids,
        profile_value_ids,
        profile_mask,
        profile_rope_time,
        history_rope_time,
        calendar_features,
    )
    loss = F.cross_entropy(
        output.logits.reshape(-1, vocabulary_size),
        masked["mlm_labels"].reshape(-1),
        ignore_index=-100,
    )
    loss.backward()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"logits:           {tuple(output.logits.shape)}")
    print(f"account embedding:{tuple(output.account_embedding.shape)}")
    print(f"event embeddings: {tuple(output.contextual_event_embeddings.shape)}")
    print(f"parameters:       {parameter_count:,}")
    print(f"MLM loss:         {loss.item():.4f}")


if __name__ == "__main__":
    main()
