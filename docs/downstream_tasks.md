# Downstream evaluation protocol

The pretrained model produces one `account_embedding` per account snapshot.
Every supervised example has an `account_id`, a `cutoff_time`, a future-derived
`target`, and its account-disjoint split. The model may only see transactions
and profile events available on or before its cutoff.

## Evaluation order

1. Fit the task definition, feature schema, and hyperparameters on training
   data; use validation only for model selection.
2. Run a prevalence/mean baseline and a tabular baseline.
3. Run the frozen embedding probe using the identical cached task table.
4. Inspect validation results. Fine-tune only if the frozen embedding is
   competitive with the tabular baseline. Keep the test set for one final
   report and include its uncertainty interval.

## Loan repayment trouble

The label is one for published loan statuses `B` or `D`. The snapshot ends one
microsecond before loan grant, excluding grant-day transactions and the
`loan_granted` profile milestone.

```bash
python scripts/run_loan_probe.py --checkpoint /path/to/best.pt
python scripts/run_loan_tabular_baseline.py \
  --output /path/to/loan_tabular_baseline_metrics.json \
  --frozen-probe-metrics /path/to/loan_probe_metrics.json
```

The current frozen probe does **not** beat the tabular baseline on the observed
test split, so do not fine-tune loan risk yet. The test also has only eight
positive examples.

## Cash-flow stress

Target: for an account currently *above* the threshold, its minimum balance
from just after the cutoff through the next 60 days crosses at or below the
10th-percentile balance fitted from training events. Cutoffs are 180 days
apart by default, producing several examples per account and requiring a
complete future window. Excluding accounts already below the threshold keeps
this a forecasting task rather than current-state detection.

```bash
python scripts/run_forward_task_baseline.py --task cashflow_stress \
  --output /path/to/cashflow_tabular_baseline.json
python scripts/run_forward_embedding_probe.py --task cashflow_stress \
  --checkpoint /path/to/best.pt \
  --task-table /path/to/cashflow_stress_task_table.parquet \
  --output /path/to/cashflow_frozen_probe.json --device cuda
```

## Future-value proxy

This is not literal LTV. Target: `log1p(sum(abs(amount)))` during the next 180
days. The task table uses the same cutoff safeguards and can be reused by the
baseline and frozen embedding probe.

```bash
python scripts/run_forward_task_baseline.py --task future_value \
  --output /path/to/future_value_tabular_baseline.json
python scripts/run_forward_embedding_probe.py --task future_value \
  --checkpoint /path/to/best.pt \
  --task-table /path/to/future_value_task_table.parquet \
  --output /path/to/future_value_frozen_probe.json --device cuda
```

The Czech data has no verified fraud labels. Any later fraud-related work here
must be described as transaction anomaly scoring, not fraud detection.
