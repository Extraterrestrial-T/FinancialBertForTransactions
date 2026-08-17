# Initial results

These are exploratory, account-disjoint Czech-bank experiments. The test set
was evaluated once after all hyperparameter and rank choices were made using
validation only. Cash-flow and future-value task tables can contain multiple
snapshots per account; uncertainty intervals therefore resample whole accounts,
not individual rows.

## Cash-flow stress: next 60 days

The target is whether an account that is currently above a low-balance
threshold fitted from the training split crosses that threshold during the
following 60 days. The held-out set contains 3,531 dated account snapshots,
including 607 stress cases (17.2%). Average precision is especially important
here because the task is imbalanced; lower Brier score means better probability
calibration.

| Model | ROC-AUC | Average precision | Brier score | Balanced accuracy |
|---|---:|---:|---:|---:|
| Tabular logistic baseline | 0.818 | 0.480 | 0.114 | 0.544 |
| Frozen Transformer + logistic head | 0.805 | 0.462 | 0.117 | 0.594 |
| History Encoder LoRA (rank 16) | **0.864** | **0.614** | **0.102** | **0.720** |

The LoRA adapter improves average precision by 0.134 over the engineered
tabular baseline and by 0.152 over the frozen embedding. Its account-clustered
95% bootstrap interval for test average precision is **[0.569, 0.657]**,
compared with **[0.438, 0.525]** for the tabular baseline. This is the strongest
classification result in the project, while still remaining an exploratory
historical stress proxy rather than a deployment claim.

## Future transaction-volume proxy: next 180 days

The target is `log1p(sum(abs(amount)))` during the next 180 days. In other
words, it is `ln(1 + future absolute transaction volume)`. This keeps zero
volume valid and compresses unusually high-volume accounts. It is **not true
LTV**: the dataset has neither revenue nor retention labels, and MAE/RMSE are
measured in log-volume units rather than currency units.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Mean-target baseline | 0.787 | 0.937 | -0.001 |
| Tabular Ridge baseline | 0.275 | 0.443 | 0.776 |
| Frozen Transformer + Ridge head | 0.261 | 0.440 | 0.780 |
| History Encoder LoRA (rank 4) | **0.223** | **0.408** | **0.811** |

The rank-4 adapter trains only 20,736 LoRA parameters—about 1.3% of the
1.59M-parameter backbone—plus its one-dimensional task head. It lowers test
MAE by about 19% versus tabular Ridge and 15% versus frozen embeddings. Its
account-clustered test MAE interval is **[0.205, 0.239]**, below tabular
Ridge's **[0.257, 0.293]**. This is the project’s primary adaptation result.

## What is intentionally not claimed

The loan-repayment diagnostic had only eight positive held-out examples and
tabular features won, so it is not a fine-tuning candidate. The Czech dataset
has no verified fraud labels, so the project makes no fraud-detection claim.
The MLM base checkpoint was not re-pretrained separately for every downstream
cutoff horizon; treat all results as representation-learning evidence, not
prospective production-risk estimates.
