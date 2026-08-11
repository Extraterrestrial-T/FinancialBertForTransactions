# Czech bank data card

## Purpose and provenance

This project uses the modified Czech financial dataset packaged in the
`dnoeth/1999_Czech_financial_dataset_Teradata` archive. It originates from the
PKDD’99 Discovery Challenge data. The local Teradata variant shifts dates
forward by twenty years, divides amounts by ten, and translates selected Czech
transaction descriptions to short English codes. It contains roughly 1.06M
transactions across 4,500 accounts, together with client, account, card, loan,
order, and district tables.

It is a historical, de-identified teaching dataset. It is not representative
of a modern bank, has no free-text transaction descriptions, and supplies no
verified fraud labels. Any claim in this repository is therefore limited to
the supplied synthetic/historical task definitions.

## Version-controlled artifacts

`data/processed/czech_bank/` is intentionally committed. It contains the
account-disjoint split manifest, split transaction/profile Parquet files, and
two compact derived artifacts:

- `lifelong_events.parquet`: dated `account_opened`, `card_issued`, and
  `loan_granted` records used to materialize profile state at a sample cutoff.
- `loan_outcomes.parquet`: loan grant time and status used only by the
  exploratory loan diagnostic.

The raw `financial_db_Teradata/` export, MBD archive, Indian transaction CSV,
environments, checkpoints, adapters, local reports, and Drive mirrors are
ignored. The processed data is small enough to reproduce the project without
shipping the original raw archive.

## Processing and split protocol

The preparer normalizes transaction dates, joins static account/client/district
information into the profile rows, and assigns every account to exactly one
train/validation/test split. Tokenizer numeric bucket edges and categorical
vocabularies are fitted from train records only. A sample’s profile state is
then materialized at its cutoff: life-long events after the cutoff are absent;
its event history includes only transactions at or before that cutoff.

To rebuild the processed artifacts after explicitly placing the raw Teradata
TSVs under `financial_db_Teradata/`, run:

```powershell
python scripts/prepare_czech_data.py `
  --raw-dir financial_db_Teradata `
  --output-dir data/processed/czech_bank
```

The command overwrites its specified output directory. Use a separate staging
path if you want to inspect a newly generated version before replacing the
committed artifacts.

## Downstream labels

Cash-flow stress and future value use multiple dated snapshots per account.
The split is still account-disjoint, and bootstrap uncertainty resamples whole
accounts so several snapshots from one account are never treated as fully
independent. Read [the downstream protocol](downstream_tasks.md) for the
label details and the rationale for excluding loan fine-tuning.
