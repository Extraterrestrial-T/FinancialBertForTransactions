"""Rebuild the committed Czech-bank processed artifacts from raw Teradata TSVs.

The public repository includes the resulting compact Parquet files. This
script is optional and exists for provenance, not as a prerequisite for model
training or notebook evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pragma_lite.data.lifelong import (  # noqa: E402
    materialize_lifelong_events,
    materialize_loan_outcomes,
    read_lifelong_source_tables,
)


TRANS_COLUMNS = [
    "trans_id", "account_id", "trans_date", "amount", "balance",
    "trans_type", "operation", "category", "other_bank_id", "other_account_id",
]
ACCOUNT_COLUMNS = ["account_id", "district_id", "create_date", "frequency"]
DISP_COLUMNS = ["disp_id", "client_id", "account_id", "disp_type"]
CLIENT_COLUMNS = ["client_id", "birth_date", "gender", "district_id"]
DISTRICT_COLUMNS = [
    "district_id", "district_name", "region", "num_inhabitants",
    "num_municipalities_gt499", "num_municipalities_500to1999",
    "num_municipalities_2000to9999", "num_municipalities_gt10000", "num_cities",
    "ratio_urban", "average_salary", "unemployment_rate95", "unemployment_rate96",
    "num_entrep_per1000", "num_crimes95", "num_crimes96",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "czech_bank")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--min-history-span-days", type=int, default=14)
    return parser.parse_args()


def _read_tsv(raw_dir: Path, name: str, columns: list[str], *, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"missing raw Czech-bank file: {path}")
    return pd.read_csv(
        path, sep="\t", header=None, names=columns, na_values=[""],
        keep_default_na=True, parse_dates=list(parse_dates), low_memory=False,
    )


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transactions = _read_tsv(raw_dir, "fin_trans.tsv", TRANS_COLUMNS, parse_dates=("trans_date",))
    accounts = _read_tsv(raw_dir, "fin_account.tsv", ACCOUNT_COLUMNS, parse_dates=("create_date",))
    dispositions = _read_tsv(raw_dir, "fin_disp.tsv", DISP_COLUMNS)
    clients = _read_tsv(raw_dir, "fin_client.tsv", CLIENT_COLUMNS, parse_dates=("birth_date",))
    districts = _read_tsv(raw_dir, "fin_district.tsv", DISTRICT_COLUMNS)

    for name in ("trans_type", "operation", "category", "other_bank_id"):
        transactions[name] = transactions[name].astype("string").str.strip().replace("", pd.NA)
    transactions = (
        transactions.dropna(subset=["account_id", "trans_date", "amount", "balance"])
        .drop_duplicates(subset=["trans_id"])
        .sort_values(["account_id", "trans_date", "trans_id"])
        .reset_index(drop=True)
    )

    history = transactions.groupby("account_id").agg(
        n_events=("trans_id", "size"), first_event=("trans_date", "min"), last_event=("trans_date", "max")
    ).reset_index()
    history["history_span_days"] = (history["last_event"] - history["first_event"]).dt.total_seconds() / 86_400
    manifest = history.loc[
        history["n_events"].ge(args.min_events)
        & history["history_span_days"].ge(args.min_history_span_days)
    ].copy()
    account_ids = manifest["account_id"].to_numpy().copy()
    np.random.default_rng(args.seed).shuffle(account_ids)
    train_end = round(len(account_ids) * 0.70)
    valid_end = train_end + round(len(account_ids) * 0.15)
    split_by_account = {
        **{account_id: "train" for account_id in account_ids[:train_end]},
        **{account_id: "valid" for account_id in account_ids[train_end:valid_end]},
        **{account_id: "test" for account_id in account_ids[valid_end:]},
    }
    manifest["split"] = manifest["account_id"].map(split_by_account)

    owners = dispositions.loc[dispositions["disp_type"].eq("O"), ["account_id", "client_id"]]
    if not owners["account_id"].is_unique:
        raise ValueError("expected one account owner per account")
    profile = (
        accounts.merge(owners, on="account_id", how="left", validate="one_to_one")
        .merge(clients.rename(columns={"district_id": "client_district_id"}), on="client_id", how="left", validate="many_to_one")
        .merge(districts, left_on="client_district_id", right_on="district_id", how="left", suffixes=("", "_district"), validate="many_to_one")
    )
    events = transactions.merge(manifest[["account_id", "split"]], on="account_id", how="inner", validate="many_to_one")
    profiles = profile.merge(manifest[["account_id", "split"]], on="account_id", how="inner", validate="one_to_one")
    for split in ("train", "valid", "test"):
        events.loc[events["split"].eq(split)].sort_values(["account_id", "trans_date", "trans_id"]).to_parquet(output_dir / f"events_{split}.parquet", index=False)
        profiles.loc[profiles["split"].eq(split)].sort_values("account_id").to_parquet(output_dir / f"profile_{split}.parquet", index=False)
    manifest.to_parquet(output_dir / "account_split_manifest.parquet", index=False)

    tables = read_lifelong_source_tables(raw_dir)
    materialize_lifelong_events(tables).to_parquet(output_dir / "lifelong_events.parquet", index=False)
    materialize_loan_outcomes(tables).to_parquet(output_dir / "loan_outcomes.parquet", index=False)
    print(f"wrote processed Czech-bank artifacts to {output_dir}")


if __name__ == "__main__":
    main()
