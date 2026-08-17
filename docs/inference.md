# Inference: bind a base checkpoint and a task adapter

A LoRA adapter is not a complete model. It contains only low-rank residual
matrices for selected History Encoder projections and the one-dimensional task
head. To make a prediction it must be combined with:

1. the exact MLM base checkpoint on which it was trained;
2. the tokenizers fitted on that base model's training split; and
3. one cutoff-safe account snapshot.

`AccountTaskPredictor` is the public contract that performs this binding. The
adapter checkpoint records the SHA-256 hash of its base checkpoint. The loader
compares that hash before it loads any adapter tensors, so an adapter cannot
silently be paired with a different backbone.

```python
from pragma_lite import AccountTaskPredictor, load_processed_czech_account_sample

predictor = AccountTaskPredictor.from_checkpoints(
    base_checkpoint_path="/models/pragma_lite_mlm/best.pt",
    adapter_checkpoint_path="/models/adapters/future_value_lora_rank_04.pt",
    device="cpu",
)

prepared = load_processed_czech_account_sample(
    "data/processed/czech_bank",
    account_id=103,
)
prediction = predictor.predict(prepared)
print(prediction.predicted_log_future_volume)
print(prediction.predicted_future_volume)  # expm1(model output), clamped to zero
```

For a new source system, normalize its transaction rows into the documented
`Transaction` fields and call `predict_from_rows`. Supply only records known
on or before the requested cutoff. The API independently filters future
events again before tokenization.

```python
prediction = predictor.predict_from_rows(
    profile_row=normalized_profile,
    lifelong_events=normalized_milestones,
    transactions=normalized_transactions,
    cutoff_time=cutoff_timestamp,
)
```

## Swapping tasks

Keep the base checkpoint and its `tokenizers/` directory together. To switch
from future-value regression to cash-flow stress classification, create a new
predictor with the same base path and a different compatible adapter path:

```python
future = AccountTaskPredictor.from_checkpoints(base_path, future_adapter)
stress = AccountTaskPredictor.from_checkpoints(base_path, stress_adapter)
```

The future-value adapter outputs `log1p` of 180-day absolute transaction
volume; the convenience result also shows `expm1` of that value. This is a
transaction-volume proxy rather than literal LTV. The cash-flow adapter
outputs a probability for the project's historical 60-day stress label. Both
are exploratory Czech-bank research outputs, not financial decisions.

## Do not confuse token buckets with downstream outputs

The event tokenizer turns each observed amount and balance into one of its
train-fitted quantile tokens. The MLM pretraining head can emit logits over
those vocabulary tokens for a masked field. The current future-volume LoRA
adapter does something different: it has a one-dimensional regression head
trained with SmoothL1 loss on a continuous `log1p(total future volume)` target.
Its result is a scalar prediction, not a vocabulary logit or an estimated
future bucket. The cash-flow adapter's scalar *is* a classification logit,
which is converted to a probability with sigmoid.

## CLI example

The repository command uses the same contract and prints one JSON prediction:

```powershell
python scripts/run_account_inference.py `
  --base-checkpoint C:\models\best.pt `
  --adapter C:\models\future_value_lora_rank_04.pt `
  --account-id 103
```

The portfolio release keeps its compact model artifacts under
`app_assets/artifacts/`. The Flask explorer discovers `best.pt`, its sibling
tokenizer state, and the curated LoRA adapters from that directory. A private
deployment can instead construct `AccountTaskPredictor` with explicit paths.
