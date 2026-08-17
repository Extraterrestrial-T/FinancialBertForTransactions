# Committed model release

This directory contains the compact, reproducible model release used by the
Flask research site. It intentionally preserves the experiment layout that
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
  lab_index.json
```

The live application finds `best.pt` by locating its sibling `tokenizers/`
directory and recursively discovers `.pt` files carrying task-adapter
metadata. `last.pt` is retained only as a pretraining-resume checkpoint;
inference and downstream LoRA use `best.pt`.

For a leaner later release, you may omit `last.pt`, the non-selected LoRA
ranks, and cached task tables. Keep `best.pt`, all three tokenizer JSON files,
the selected rank-4 future-value adapter, the selected rank-16 cash-flow
adapter, and their reports.

# Deployment artifacts

`lab_index.json` is a compact, precomputed list of held-out accounts and their
valid cutoff dates. It is derived from the committed cached task tables with:

```powershell
./FinBert/Scripts/python.exe scripts/build_lab_index.py --model-dir app_assets/artifacts/models/checkpoints_swoll/pragma_lite_mlm --output app_assets/artifacts/lab_index.json
```

The Flask app reads this small file for the account picker instead of scanning
the full transaction history on a constrained deployment instance.
