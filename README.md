# PRAGMA-lite

A compact experiment in learning account representations from
structured banking histories. It mirrors the useful ideas in
[PRAGMA](https://arxiv.org/html/2604.08649v1) at a deliberately smaller
scale: field-pair tokenisation, event/profile/history encoders, time-aware
attention, masked financial-event modelling, and parameter-efficient
adaptation for forward-looking tasks.

This is a learning and research project, not a credit, fraud, or production
risk model. The code is organised so the notebooks tell the story while the
installable package contains the reusable implementation.

## What is here

- A committed, processed Czech-bank split with account-disjoint train,
  validation, and test accounts. Raw Teradata exports are deliberately local.
- A small three-stage backbone: an Event Encoder turns structured transaction
  fields into event vectors; a Profile Encoder summarizes static and
  life-long account state; a History Encoder contextualizes the event stream.
- Masked value modelling that updates all three encoders through the History
  Encoder’s contextual event representations.
- Leakage-safe downstream task tables, engineered tabular baselines, frozen
  embedding probes, and History Encoder-only LoRA adaptation.

The current exploratory evidence makes a useful distinction. Loan repayment
trouble is retained as a diagnostic only: its test set is very small and
engineered tabular features outperform the frozen representation. Cash-flow
stress shows that frozen embeddings are competitive with engineered features.
The 180-day future transaction-volume proxy is the primary adaptation task:
the frozen representation slightly exceeded the tabular Ridge baseline in the
initial held-out experiment, and History Encoder LoRA improves it further. The
target is ``log1p(sum(abs(amount)))`` in the next 180 days: ``log1p(x)`` is
``ln(1 + x)``, which safely handles zero volume and compresses extreme account
activity. It is a volume proxy, not true customer LTV; its MAE is on the
log-volume scale, not in currency units.

## Repository map

```text
src/pragma_lite/
  data/       typed records, tokenizers, processed-data dataset and collation
  models/     manual attention, Event/Profile/History encoders, MLM backbone
  tasks/      cutoff-safe labels, tabular features, metrics and uncertainty
  training/   MLM loop plus History Encoder LoRA adaptation
scripts/      runnable commands used by the notebooks
docs/         data card, architecture, downstream protocol and experiment story
01_*.ipynb … 05_*.ipynb  reader-facing notebook workflow
data/processed/czech_bank/  committed reproducible data artifacts
```

## Local setup

Python 3.11+ is required. Create an environment, then install this project in
editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,test]"
python -m unittest discover -s tests -v
```

To run the interactive explorer, add its small web extra:

```powershell
python -m pip install -e ".[web]"
python -m flask --app webapp.server run --debug
```

The processed artifacts let you run tests, data inspection, frozen evaluation,
and smoke training without the raw source archive. To recreate the processed
files from a local Teradata export, see [the data card](docs/data.md).

## Colab: local SSD for work, Drive for persistence

Connecting a VS Code notebook to Colab executes the notebook kernel on a
temporary Colab VM. A persistent connection keeps that runtime alive only
while the session remains active; it does not turn Drive into the VM’s normal
filesystem. Clone and install under `/content` so code and Parquet I/O use the
fast local SSD. Mount Drive only for artifacts you need after the session
ends: checkpoints, cached task tables, reports, and LoRA adapters.

```python
%cd /content
!git clone https://github.com/Extraterrestrial-T/FinancialBertForTransactions.git
%cd /content/FinancialBertForTransactions
!pip -q install -e ".[notebooks]"

from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")
PERSIST_ROOT = Path("/content/drive/MyDrive/FinancialBertForTransactions")
CHECKPOINT_DIR = PERSIST_ROOT / "checkpoints" / "pragma_lite_mlm"
REPORT_DIR = PERSIST_ROOT / "reports"
ADAPTER_DIR = PERSIST_ROOT / "adapters"
for directory in (CHECKPOINT_DIR, REPORT_DIR, ADAPTER_DIR):
    directory.mkdir(parents=True, exist_ok=True)
```

The notebooks use these locations intentionally: models read source data from
the clone, while only small durable artifacts are written to Drive.

## Notebook path

Run the notebooks in order. They are deliberately light on implementation
details and call into `pragma_lite` or `scripts/` instead.

1. `01_data_and_splits.ipynb` — provenance, EDA decisions, account split, and
   optional artifact rebuilding.
2. `02_pretrain_mlm.ipynb` — MLM pre-training and a 256-event context run.
3. `03_frozen_downstream_evaluation.ipynb` — task tables, tabular baselines,
   frozen transfer probes, and interpretation.
4. `04_lora_cashflow_stress.ipynb` — rank-selected classification adaptation.
5. `05_lora_future_value.ipynb` — rank-selected regression adaptation.

Read [the architecture note](docs/architecture.md),
[the downstream protocol](docs/downstream_tasks.md), and
[the experiment story](docs/experiment_story.md) before presenting results.
The compact, recruiter-facing numbers and their boundaries live in
[the results note](docs/results.md). For exact base-checkpoint plus LoRA
adapter loading, see [the inference contract](docs/inference.md).

## Interactive explorer

An web app  explorer lives in [`webapp/`](webapp/). It has
a click-driven transaction and balance timeline rather than a notebook-style
dashboard: choose one adapter, anonymous account and cutoff, run real
inference, then inspect the model-visible events alongside the held-out
outcome. It verifies the base-plus-adapter contract before each model is
loaded. The included [Vercel configuration](vercel.json) deploys the same
application without moving this into a separate repository. Read [the web app
guide](webapp/README.md) and [the model-release layout](app_assets/artifacts/README.md).

## Scope and caveats

The Czech dataset has dated, structured transactions but no text descriptions
and no verified fraud labels. This repository makes no fraud-detection claim.
All forward-looking evaluations use account-disjoint splits and cutoff-safe
input histories, but the current MLM checkpoint was pretrained on full
account histories rather than being re-pretrained per downstream horizon.
