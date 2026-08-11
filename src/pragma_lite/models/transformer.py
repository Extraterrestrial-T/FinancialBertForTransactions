"""Manual Transformer primitives and a small PRAGMA-lite Event Encoder.

This module deliberately implements attention rather than wrapping
``torch.nn.MultiheadAttention``.  It is the reusable foundation for the
future Profile State and History Encoders, while the runnable model here only
covers transaction-event MLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Hyperparameters shared by the attention blocks in this experiment."""

    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 128
    dropout: float = 0.1
    rope_base: float | None = 10_000.0

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.ffn_dim <= 0 or self.num_layers <= 0:
            raise ValueError("d_model, ffn_dim, and num_layers must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_base is not None:
            if self.rope_base <= 1.0:
                raise ValueError("rope_base must be greater than one")
            if (self.d_model // self.num_heads) % 2:
                raise ValueError("RoPE requires an even per-head dimension")


def apply_rotary_embedding(
    vectors: Tensor,
    positions: Tensor,
    *,
    base: float = 10_000.0,
) -> Tensor:
    """Apply RoPE to Q or K vectors with continuous token coordinates.

    Args:
        vectors: Tensor shaped ``[batch, heads, sequence, head_dim]``.
        positions: Continuous positions shaped ``[sequence]`` or
            ``[batch, sequence]``.  PRAGMA's log-time coordinates may be
            passed directly; they need not be integer token positions.
        base: RoPE frequency base.
    """
    if vectors.ndim != 4:
        raise ValueError("vectors must have shape [batch, heads, sequence, head_dim]")
    batch_size, _, sequence_length, head_dim = vectors.shape
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    if base <= 1.0:
        raise ValueError("base must be greater than one")

    if positions.ndim == 1:
        if positions.shape[0] != sequence_length:
            raise ValueError("one-dimensional positions must match sequence length")
        positions = positions.unsqueeze(0).expand(batch_size, -1)
    elif positions.ndim == 2:
        if tuple(positions.shape) != (batch_size, sequence_length):
            raise ValueError("positions must have shape [batch, sequence]")
    else:
        raise ValueError("positions must have shape [sequence] or [batch, sequence]")

    # Compute the angles in float32 for stable trigonometry, then return to
    # the activation dtype so this remains usable with future mixed precision.
    position_values = positions.to(device=vectors.device, dtype=torch.float32)
    frequency_indices = torch.arange(
        0, head_dim, 2, device=vectors.device, dtype=torch.float32
    )
    inverse_frequencies = 1.0 / (base ** (frequency_indices / head_dim))
    angles = position_values[:, None, :, None] * inverse_frequencies[None, None, None, :]
    cosines = torch.cos(angles).to(dtype=vectors.dtype)
    sines = torch.sin(angles).to(dtype=vectors.dtype)

    even = vectors[..., 0::2]
    odd = vectors[..., 1::2]
    rotated_pairs = torch.stack(
        (even * cosines - odd * sines, even * sines + odd * cosines), dim=-1
    )
    return rotated_pairs.flatten(start_dim=-2)
    

class MultiHeadAttention(nn.Module):
    """Bidirectional masked multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        rope_base: float | None = 10_000.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_base = rope_base
        if rope_base is not None:
            if rope_base <= 1.0:
                raise ValueError("rope_base must be greater than one")
            if self.head_dim % 2:
                raise ValueError("RoPE requires an even per-head dimension")

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / sqrt(self.head_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        *,
        rope_positions: Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Attend over ``x`` while excluding ``False`` entries in the mask.

        ``attention_mask`` is shaped ``[batch, sequence]`` and uses ``True``
        for real tokens.  This is intentionally non-causal because masked
        language modelling can use both left and right context.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, sequence, d_model]")
        batch_size, sequence_length, hidden_size = x.shape
        if hidden_size != self.d_model:
            raise ValueError(f"expected hidden size {self.d_model}, got {hidden_size}")
        valid_tokens = self._normalise_attention_mask(
            attention_mask, batch_size, sequence_length, x.device
        )

        queries = self._split_heads(self.q_proj(x))
        keys = self._split_heads(self.k_proj(x))
        values = self._split_heads(self.v_proj(x))
        if rope_positions is not None:
            if self.rope_base is None:
                raise ValueError("this attention module was created with RoPE disabled")
            queries = apply_rotary_embedding(queries, rope_positions, base=self.rope_base)
            keys = apply_rotary_embedding(keys, rope_positions, base=self.rope_base)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~valid_tokens[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # If every key is padded, softmax(-inf, ..., -inf) is NaN.  Such a row
        # should contribute zero rather than contaminate an entire batch.
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = self.attention_dropout(weights)

        attended = torch.matmul(weights, values)
        output = self.out_proj(self._combine_heads(attended))
        output = self.output_dropout(output)
        output = output * valid_tokens.unsqueeze(-1).to(dtype=output.dtype)
        if return_attention_weights:
            return output, weights
        return output

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        return x.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        batch_size, _, sequence_length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)

    @staticmethod
    def _normalise_attention_mask(
        attention_mask: Tensor | None,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones((batch_size, sequence_length), dtype=torch.bool, device=device)
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask must have shape [batch, sequence]")
        return attention_mask.to(device=device, dtype=torch.bool)


class FeedForwardNetwork(nn.Module):
    """Position-wise Transformer MLP: Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, d_model: int, ffn_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0 or ffn_dim <= 0:
            raise ValueError("d_model and ffn_dim must be positive")
        self.input_proj = nn.Linear(d_model, ffn_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(ffn_dim, d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_dropout(self.output_proj(self.dropout(self.activation(self.input_proj(x)))))


class TransformerBlock(nn.Module):
    """Pre-norm attention and feed-forward residual block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            rope_base=config.rope_base,
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForwardNetwork(
            config.d_model, config.ffn_dim, dropout=config.dropout
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        *,
        rope_positions: Tensor | None = None,
    ) -> Tensor:
        """Transform real tokens and keep padded-token vectors exactly zero."""
        valid_tokens = MultiHeadAttention._normalise_attention_mask(
            attention_mask, x.shape[0], x.shape[1], x.device
        )
        x = x + self.attention(
            self.attention_norm(x), valid_tokens, rope_positions=rope_positions
        )
        x = x * valid_tokens.unsqueeze(-1).to(dtype=x.dtype)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x * valid_tokens.unsqueeze(-1).to(dtype=x.dtype)


@dataclass(frozen=True, slots=True)
class EventEncoderOutput:
    """Contextual event summary and field vectors returned by EventEncoder."""

    event_embeddings: Tensor #[Batch, Events(E), d_model]
    field_embeddings: Tensor


class EventEncoder(nn.Module):
    """Encode every transaction independently across its structured fields.

    The same Transformer blocks are applied to every event.  A learned [EVT]
    vector is prepended internally, and its final vector becomes the event
    representation that the future History Encoder will consume.
    """

    def __init__(self, vocabulary_size: int, config: TransformerConfig, *, pad_id: int = 0) -> None:
        super().__init__()
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if not 0 <= pad_id < vocabulary_size:
            raise ValueError("pad_id must be a valid vocabulary ID")
        self.vocabulary_size = vocabulary_size
        self.config = config
        self.key_embedding = nn.Embedding(vocabulary_size, config.d_model, padding_idx=pad_id)
        self.value_embedding = nn.Embedding(vocabulary_size, config.d_model, padding_idx=pad_id)
        self.event_token = nn.Parameter(torch.empty(1, 1, 1, config.d_model))
        nn.init.normal_(self.event_token, mean=0.0, std=0.02)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))

    def forward(
        self,
        event_key_ids: Tensor,
        event_value_ids: Tensor,
        event_mask: Tensor,
    ) -> EventEncoderOutput:
        """Return event summaries and contextual field vectors.

        Args:
            event_key_ids: Field-key IDs shaped ``[B, E, F]``.
            event_value_ids: Field-value IDs shaped ``[B, E, F]``.
            event_mask: ``True`` for real events, shaped ``[B, E]``.
        """
        self._validate_inputs(event_key_ids, event_value_ids, event_mask)
        batch_size, event_count, field_count = event_key_ids.shape
        event_mask = event_mask.to(device=event_key_ids.device, dtype=torch.bool)
        field_mask = event_mask.unsqueeze(-1).expand(-1, -1, field_count)

        field_vectors = self.key_embedding(event_key_ids) + self.value_embedding(event_value_ids)
        field_vectors = self.embedding_dropout(field_vectors)
        summary_vectors = self.event_token.expand(batch_size, event_count, 1, -1)
        vectors = torch.cat((summary_vectors, field_vectors), dim=2)
        token_mask = torch.cat((event_mask.unsqueeze(-1), field_mask), dim=2)
        vectors = vectors * token_mask.unsqueeze(-1).to(dtype=vectors.dtype)

        tokens_per_event = field_count + 1
        vectors = vectors.reshape(batch_size * event_count, tokens_per_event, self.config.d_model)
        token_mask = token_mask.reshape(batch_size * event_count, tokens_per_event)
        for block in self.blocks:
            vectors = block(vectors, token_mask)

        vectors = vectors.reshape(batch_size, event_count, tokens_per_event, self.config.d_model)
        return EventEncoderOutput(
            event_embeddings=vectors[:, :, 0],
            field_embeddings=vectors[:, :, 1:],
        )

    @staticmethod
    def _validate_inputs(
        event_key_ids: Tensor, event_value_ids: Tensor, event_mask: Tensor
    ) -> None:
        if event_key_ids.ndim != 3:
            raise ValueError("event_key_ids must have shape [batch, events, fields]")
        if event_value_ids.shape != event_key_ids.shape:
            raise ValueError("event_value_ids must have the same shape as event_key_ids")
        if event_mask.shape != event_key_ids.shape[:2]:
            raise ValueError("event_mask must have shape [batch, events]")


@dataclass(frozen=True, slots=True)
class ProfileEncoderOutput:
    """Contextual user summary and profile-pair embeddings."""

    user_embedding: Tensor  # [B, D]
    field_embeddings: Tensor  # [B, P, D]


class ProfileEncoder(nn.Module):
    """Encode static fields and time-filtered life-long profile events."""

    def __init__(self, vocabulary_size: int, config: TransformerConfig, *, pad_id: int = 0) -> None:
        super().__init__()
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if not 0 <= pad_id < vocabulary_size:
            raise ValueError("pad_id must be a valid vocabulary ID")
        self.vocabulary_size = vocabulary_size
        self.config = config
        self.key_embeddings = nn.Embedding(vocabulary_size, config.d_model, padding_idx=pad_id)
        self.value_embeddings = nn.Embedding(vocabulary_size, config.d_model, padding_idx=pad_id)
        self.user_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        nn.init.normal_(self.user_token, mean=0.0, std=0.02)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))

    def forward(
        self,
        profile_key_ids: Tensor,
        profile_value_ids: Tensor,
        profile_mask: Tensor,
        profile_rope_time: Tensor,
    ) -> ProfileEncoderOutput:
        """Return the learned ``[USR]`` vector and contextual profile fields.

        Dated life-long events are already filtered to the sample cutoff by
        the data handler. Their log-time coordinates let RoPE distinguish
        recent milestones from account-opening information.
        """
        self._validate_inputs(profile_key_ids, profile_value_ids, profile_mask)
        batch_size, _ = profile_key_ids.shape
        profile_mask = profile_mask.to(device=profile_key_ids.device, dtype=torch.bool)
        if profile_rope_time.shape != profile_key_ids.shape:
            raise ValueError("profile_rope_time must have shape [batch, fields]")
        profile_rope_time = profile_rope_time.to(device=profile_key_ids.device, dtype=torch.float32)

        field_vectors = self.key_embeddings(profile_key_ids) + self.value_embeddings(profile_value_ids)
        field_vectors = self.embedding_dropout(field_vectors)
        vectors = torch.cat((self.user_token.expand(batch_size, 1, -1), field_vectors), dim=1)
        token_mask = torch.cat(
            (
                torch.ones((batch_size, 1), dtype=torch.bool, device=profile_key_ids.device),
                profile_mask,
            ),
            dim=1,
        )
        positions = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=profile_rope_time.dtype, device=profile_key_ids.device),
                profile_rope_time,
            ),
            dim=1,
        )
        vectors = vectors * token_mask.unsqueeze(-1).to(dtype=vectors.dtype)
        for block in self.blocks:
            vectors = block(vectors, attention_mask=token_mask, rope_positions=positions)
        return ProfileEncoderOutput(user_embedding=vectors[:, 0], field_embeddings=vectors[:, 1:])

    @staticmethod
    def _validate_inputs(
        profile_key_ids: Tensor, profile_value_ids: Tensor, profile_mask: Tensor
    ) -> None:
        if profile_key_ids.ndim != 2:
            raise ValueError("profile_key_ids must have shape [batch, fields]")
        if profile_value_ids.shape != profile_key_ids.shape:
            raise ValueError("profile_value_ids must have the same shape as profile_key_ids")
        if profile_mask.shape != profile_key_ids.shape[:2]:
            raise ValueError("profile_mask must have shape [batch, fields]")

@dataclass(frozen=True, slots=True)
class HistoryEncoderOutput:
    """Output of the History Encoder, including a user embedding and contextual event embeddings."""

    account_embedding: Tensor #[Batch, d_model]
    contextual_event_embeddings: Tensor #[Batch, Events(E), d_model]


class HistoryEncoder(nn.Module):
    """Contextualise an account's profile summary and event history.

    This encoder deliberately has no vocabulary embedding tables: its inputs
    are dense vectors produced by :class:`ProfileEncoder` and
    :class:`EventEncoder`.  The first position is the contextual profile
    ``[USR]`` vector; subsequent positions are transaction ``[EVT]`` vectors.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        # The data handler emits sin/cos pairs for day-of-week and
        # day-of-month.  Calendar information belongs to event positions,
        # rather than the profile-derived [USR] position.
        self.calendar_projection = nn.Linear(4, config.d_model, bias=False)
        
    def forward(
            self,
            history_tokens: Tensor,
            event_mask: Tensor,
            calendar_features: Tensor,
            history_rope_time: Tensor,
        ) -> HistoryEncoderOutput:
        """Return the account vector and context-aware event vectors.

        Args:
            history_tokens: ``[B, E + 1, D]``.  Position zero is the profile
                ``[USR]`` vector; positions one through ``E`` are event
                summary vectors.
            event_mask: ``True`` for real events, shaped ``[B, E]``.
            calendar_features: Four continuous calendar values per event,
                shaped ``[B, E, 4]``.
            history_rope_time: PRAGMA log-time coordinates shaped ``[B, E]``.
        """
        if history_tokens.ndim != 3:
            raise ValueError("history_tokens must have shape [batch, events + 1, d_model]")
        batch_size, sequence_length, d_model = history_tokens.shape
        if d_model != self.config.d_model:
            raise ValueError(f"expected d_model {self.config.d_model}, got {d_model}")
        if sequence_length < 1:
            raise ValueError("history_tokens must contain at least the [USR] position")

        event_count = sequence_length - 1
        if event_mask.shape != (batch_size, event_count):
            raise ValueError("event_mask must have shape [batch, events]")
        if calendar_features.shape != (batch_size, event_count, 4):
            raise ValueError("calendar_features must have shape [batch, events, 4]")
        if history_rope_time.shape != (batch_size, event_count):
            raise ValueError("history_rope_time must have shape [batch, events]")

        event_mask = event_mask.to(device=history_tokens.device, dtype=torch.bool)
        calendar_features = calendar_features.to(
            device=history_tokens.device,
            dtype=self.calendar_projection.weight.dtype,
        )
        history_rope_time = history_rope_time.to(
            device=history_tokens.device,
            dtype=torch.float32,
        )

        calendar_vectors = self.calendar_projection(calendar_features).to(
            dtype=history_tokens.dtype
        )
        event_vectors = history_tokens[:, 1:] + calendar_vectors
        event_vectors = event_vectors * event_mask.unsqueeze(-1).to(dtype=event_vectors.dtype)

        x = torch.cat((history_tokens[:, :1], event_vectors), dim=1)
        history_mask = torch.cat(
            (
                torch.ones((batch_size, 1), dtype=torch.bool, device=history_tokens.device),
                event_mask,
            ),
            dim=1,
        )
        history_positions = torch.cat(
            (
                torch.zeros(
                    (batch_size, 1),
                    dtype=history_rope_time.dtype,
                    device=history_tokens.device,
                ),
                history_rope_time,
            ),
            dim=1,
        )
        x = x * history_mask.unsqueeze(-1).to(dtype=x.dtype)

        for block in self.blocks:
            x = block(x, attention_mask=history_mask, rope_positions=history_positions)

        return HistoryEncoderOutput(
            account_embedding=x[:, 0],
            contextual_event_embeddings=x[:, 1:],
        )




@dataclass(frozen=True, slots=True)
class EventMLMOutput:
    """Outputs of the end-to-end profile, event, history, and MLM path."""

    logits: Tensor
    account_embedding: Tensor
    event_embeddings: Tensor
    contextual_event_embeddings: Tensor
    field_embeddings: Tensor
    contextual_field_embeddings: Tensor


class EventMLMDemoModel(nn.Module):
    """Small end-to-end PRAGMA-style masked-event model.

    It is intentionally limited to the transaction-value MLM objective, but
    the loss now flows through the Event Encoder, Profile Encoder, and History
    Encoder.  It does not implement downstream heads or a profile MLM task.
    """

    def __init__(
        self,
        vocabulary_size: int,
        config: TransformerConfig | None = None,
        *,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = TransformerConfig() if config is None else config
        self.event_encoder = EventEncoder(vocabulary_size, self.config, pad_id=pad_id)
        self.profile_encoder = ProfileEncoder(vocabulary_size, self.config, pad_id=pad_id)
        self.history_encoder = HistoryEncoder(self.config)
        self.mlm_head = nn.Linear(self.config.d_model, vocabulary_size)

    def forward(
        self,
        event_key_ids: Tensor,
        event_value_ids: Tensor,
        event_mask: Tensor,
        profile_key_ids: Tensor,
        profile_value_ids: Tensor,
        profile_mask: Tensor,
        profile_rope_time: Tensor,
        history_rope_time: Tensor,
        calendar_features: Tensor,
    ) -> EventMLMOutput:
        encoded = self.event_encoder(event_key_ids, event_value_ids, event_mask)
        encoded_profile = self.profile_encoder(
            profile_key_ids,
            profile_value_ids,
            profile_mask,
            profile_rope_time,
        )
        history_tokens = torch.cat(
            (encoded_profile.user_embedding.unsqueeze(1), encoded.event_embeddings), dim=1
        )
        encoded_history = self.history_encoder(
            history_tokens,
            event_mask,
            calendar_features,
            history_rope_time,
        )

        # Every field receives its local event context plus its account-wide,
        # temporally contextualised event vector.  This is the bridge that
        # lets an MLM loss update all three encoders.
        contextual_field_embeddings = (
            encoded.field_embeddings
            + encoded_history.contextual_event_embeddings.unsqueeze(2)
        )
        contextual_field_embeddings = contextual_field_embeddings * event_mask.to(
            device=contextual_field_embeddings.device,
            dtype=contextual_field_embeddings.dtype,
        ).unsqueeze(-1).unsqueeze(-1)
        return EventMLMOutput(
            logits=self.mlm_head(contextual_field_embeddings),
            account_embedding=encoded_history.account_embedding,
            event_embeddings=encoded.event_embeddings,
            contextual_event_embeddings=encoded_history.contextual_event_embeddings,
            field_embeddings=encoded.field_embeddings,
            contextual_field_embeddings=contextual_field_embeddings,
        )


class TransformerModel(EventMLMDemoModel):
    """Compatibility alias for the complete PRAGMA-lite MLM backbone.

    New code should use :class:`EventMLMDemoModel`, whose name is retained
    from the project’s initial runnable attention demo.
    """
