# PRAGMA-lite web explorer

This is the recruiter-facing delivery surface for the project. It is a small
Flask application: the browser owns the visual exploration, while the server
owns artifact discovery, cutoff-safe data loading and real model inference.

The site is deliberately split into separate pages:

- `/` gives a short project index;
- `/architecture` explains the PRAGMA-inspired backbone with a Mermaid diagram;
- `/methodology` records sampling, MLM, LoRA, data quirks, and the real loss curve;
- `/results` explains the downstream comparisons and why the lab uses test accounts only;
- `/lab` scores held-out Czech-bank account windows;
- `/upload` accepts a one-account, schema-aligned CSV for non-persistent custom inference.

## Local run

```powershell
python -m pip install -e ".[web]"
python -m flask --app webapp.server run --debug
```

Open the local address Flask prints. The application discovers its release
under `app_assets/artifacts/`:

- `models/.../best.pt` and its sibling `tokenizers/` directory are the base;
- a chosen `adaptors/.../*.pt` file supplies the compatible LoRA and task head;
- the adapter's saved base-checkpoint hash is checked before attachment;
- `data/processed/czech_bank/` provides only anonymous, cutoff-safe examples.

The `last.pt` file is not used by the explorer. It is an MLM-resume state;
`best.pt` is the checkpoint selected for downstream experiments.

## Input windows and uploads

The inference lab can use either all transactions through a selected cutoff or
a user-selected start date. Training sampled random cutoffs and naturally
produced variable lengths, but the checkpoint has a fixed maximum of 256
transactions. Longer windows are therefore reduced to their most recent 256
events.

The upload page accepts a CSV with `trans_date`, `amount`, `balance`, and
`trans_type`, plus optional `operation`, `category`, `other_bank_id`,
`account_id`, and `create_date`. The application reads uploads only in memory;
it does not persist them. Custom categories not seen in Czech training map to
the unknown token, so this is a generalisation inspection tool rather than an
external validation claim.

## Render

`render.yaml` is included. Push the repository to GitHub, create a new Render
Blueprint from that repository, and let it use the included configuration.
It explicitly selects Render's Free instance type for a portfolio demo. Free
web services sleep after inactivity, so the first visitor after a quiet period
can see a short cold start. The service deliberately uses one worker: each
worker loads its own PyTorch model, so one keeps the compact model release and
cached data from being duplicated in memory. The shipped artifact release is
suitable for a public portfolio demo only because its records are anonymous
research data.
