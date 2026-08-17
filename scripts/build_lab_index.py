"""Build the small cutoff index served by the public inference lab.

The web app must not aggregate every transaction just to fill an account picker.
This script derives the exact test-account cutoffs from the cached task tables
that were already used for downstream evaluation, then writes a compact JSON
artifact for the deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_TABLES = {
    "cashflow_stress": "cashflow_stress_task_table.parquet",
    "future_value": "future_value_task_table.parquet",
}


def build_lab_index(model_dir: Path) -> dict[str, Any]:
    """Return task/account/cutoff data for the held-out inference lab."""
    tasks: dict[str, dict[str, Any]] = {}
    for task, filename in TASK_TABLES.items():
        table = pd.read_parquet(
            model_dir / filename,
            columns=[
                "account_id",
                "cutoff_time",
                "split",
                "history_transaction_count",
                *( ["low_balance_threshold"] if task == "cashflow_stress" else [] ),
            ],
        )
        table = table.loc[table["split"].eq("test")].copy()
        if table.empty:
            raise ValueError(f"{filename} has no test examples")
        table["cutoff_time"] = pd.to_datetime(table["cutoff_time"])
        accounts: list[dict[str, Any]] = []
        for account_id, group in table.groupby("account_id", sort=True):
            ordered = group.sort_values("cutoff_time")
            record: dict[str, Any] = {
                "account_id": int(account_id),
                "split": "test",
                "eligible_history_events": int(ordered["history_transaction_count"].max()),
                "cutoffs": [value.isoformat() for value in ordered["cutoff_time"].drop_duplicates()],
            }
            accounts.append(record)
        task_data: dict[str, Any] = {"accounts": accounts}
        if task == "cashflow_stress":
            thresholds = table["low_balance_threshold"].dropna().unique()
            if len(thresholds) != 1:
                raise ValueError("cash-flow table must contain one fitted low-balance threshold")
            task_data["low_balance_threshold"] = float(thresholds[0])
        tasks[task] = task_data
    return {"schema_version": 1, "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_lab_index(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} for {sum(len(task['accounts']) for task in payload['tasks'].values())} task-account entries.")


if __name__ == "__main__":
    main()
