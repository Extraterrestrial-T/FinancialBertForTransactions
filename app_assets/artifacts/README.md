# Committed model release

This directory contains the compact, reproducible model release used by the
Streamlit explorer. It intentionally preserves the experiment layout that
created the reported results:

```text
app_assets/artifacts/
  adaptors/
    cashflow_stress/
      cashflow_stress_lora_rank_04.pt
      cashflow_stress_lora_rank_08.pt
      cashflow_stress_lora_rank_16.pt
    future_value/
      future_value_lora_rank_04.pt
      future_value_lora_rank_08.pt
      future_value_lora_rank_16.pt
  models/checkpoints_swoll/pragma_lite_mlm/
    best.pt
    last.pt
    tokenizers/
  reports/
    cashflow_stress_lora_report.json
    future_value_lora_report.json
```

The live application finds `best.pt` by locating its sibling `tokenizers/`
directory and recursively discovers `.pt` files carrying task-adapter
metadata. `last.pt` is retained only as a pretraining-resume checkpoint;
inference and downstream LoRA use `best.pt`.

For a leaner later release, you may omit `last.pt`, the non-selected LoRA
ranks, and cached task tables. Keep `best.pt`, all three tokenizer JSON files,
the selected rank-4 future-value adapter, the selected rank-16 cash-flow
adapter, and their reports.
