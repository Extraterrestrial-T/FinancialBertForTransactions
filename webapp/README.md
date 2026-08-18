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

## Vercel

`vercel.json` and `api/index.py` deploy this same Flask application as one
Python Function. Import the GitHub repository in Vercel and leave the project
root as the repository root; no persistent volume or environment variables are
required. The function bundle explicitly includes the Flask templates, package
source, compact model release, and processed Czech demo data.

Flask is a runtime dependency in the main project dependency list because
Vercel installs that list directly; the optional `web` extra is only needed
for Gunicorn when using a conventional long-running host.

The Vercel build uses uv and is configured to resolve PyTorch from PyTorch's
CPU-only wheel index. The model runs CPU inference only; installing CUDA wheels
would add several gigabytes of NVIDIA libraries that Vercel cannot use.

During the Vercel build, `scripts/prepare_vercel_static.py` copies the browser
assets from `webapp/static/` into `public/static/`. Vercel serves those files
from its CDN while the single Flask Function handles all application and API
routes.

Vercel Functions have ephemeral local storage, which is fine here: model
artifacts and anonymous demo data are read-only bundled files, uploaded CSVs
are parsed in memory, and Python's in-process caches are only optimisations.
The initial request can be slower because it imports PyTorch and loads one
adapter on demand. If a deployment fails at build time, inspect the reported
uncompressed function size; Python functions have a bundle-size limit, so do
not commit raw data, virtual environments, or unrelated checkpoints.
