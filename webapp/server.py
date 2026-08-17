"""HTTP application behind the clickable PRAGMA-lite model explorer.

This intentionally exposes a small, fixed API over the committed anonymous
Czech-bank research artifacts.  It does not accept arbitrary checkpoint paths,
rows, or uploaded files.  That keeps the deployed demo honest about which
model and data it is showing.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from io import StringIO
import json
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from flask import Flask, jsonify, render_template, request

from pragma_lite import (
    AccountTaskPredictor,
    DemoSnapshot,
    load_demo_account_index,
    load_demo_snapshot,
    task_spec,
    transaction_bucket_label,
    valid_demo_cutoffs,
)
from pragma_lite.artifacts import AdapterArtifact, discover_base_checkpoint, discover_task_adapters


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "app_assets" / "artifacts"
PROCESSED_DIR = ROOT / "data" / "processed" / "czech_bank"
RESULTS_PATH = ROOT / "app_assets" / "initial_results.json"
DemoTask = Literal["cashflow_stress", "future_value"]

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def _valid_task(value: str | None) -> DemoTask:
    if value not in {"cashflow_stress", "future_value"}:
        raise ValueError("task must be 'cashflow_stress' or 'future_value'")
    return value  # type: ignore[return-value]


@lru_cache(maxsize=1)
def _results() -> dict[str, Any]:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _base_checkpoint() -> Path:
    checkpoint = discover_base_checkpoint(ARTIFACTS_DIR)
    if checkpoint is None:
        raise RuntimeError("Could not find best.pt and its three tokenizer-state files.")
    return checkpoint


@lru_cache(maxsize=1)
def _adapters() -> tuple[AdapterArtifact, ...]:
    return discover_task_adapters(ARTIFACTS_DIR)


@lru_cache(maxsize=2)
def _predictor(adapter_identifier: str) -> AccountTaskPredictor:
    adapter = _adapter_by_identifier(adapter_identifier)
    return AccountTaskPredictor.from_checkpoints(_base_checkpoint(), adapter.path, device="cpu")


@lru_cache(maxsize=2)
def _account_index(task: DemoTask) -> tuple[dict[str, Any], ...]:
    index = load_demo_account_index(PROCESSED_DIR, task)
    return tuple(
        {
            "account_id": int(row.account_id),
            "split": str(row.split),
            "eligible_history_events": int(row.eligible_history_events),
        }
        for row in index.loc[index["split"].eq("test")].itertuples(index=False)
    )


@lru_cache(maxsize=512)
def _cutoffs(task: DemoTask, account_id: int) -> tuple[str, ...]:
    return tuple(value.isoformat() for value in valid_demo_cutoffs(PROCESSED_DIR, task, account_id))


def _adapter_by_identifier(identifier: str) -> AdapterArtifact:
    for adapter in _adapters():
        if adapter.identifier == identifier:
            return adapter
    raise ValueError("adapter is not in this curated release")


@app.errorhandler(ValueError)
def _invalid_request(error: ValueError):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(RuntimeError)
def _artifact_problem(error: RuntimeError):
    return jsonify({"error": str(error)}), 503


@app.get("/")
def index():
    return render_template("project_home.html")


@app.get("/architecture")
def architecture():
    return render_template("architecture.html")


@app.get("/methodology")
def methodology():
    return render_template("methodology.html")


@app.get("/results")
def results():
    return render_template("results.html")


@app.get("/lab")
def lab():
    """Serve the focused interactive inference experience."""
    return render_template("inference_lab.html")


@app.get("/upload")
def upload():
    return render_template("upload.html")


@app.get("/api/bootstrap")
def bootstrap():
    adapters = [
        {
            "id": adapter.identifier,
            "task": adapter.task,
            "label": adapter.label,
            "rank": adapter.rank,
        }
        for adapter in _adapters()
    ]
    return jsonify(
        {
            "results": _results(),
            "adapters": adapters,
            "base_checkpoint": _base_checkpoint().name,
            "tasks": {
                task: {
                    "label": spec.label,
                    "horizon_days": spec.horizon_days,
                    "min_history_transactions": spec.min_history_transactions,
                }
                for task, spec in (("cashflow_stress", task_spec("cashflow_stress")), ("future_value", task_spec("future_value")))
            },
        }
    )


@app.get("/api/accounts")
def accounts():
    task = _valid_task(request.args.get("task"))
    return jsonify({"task": task, "accounts": _account_index(task)})


@app.get("/api/cutoffs")
def cutoffs():
    task = _valid_task(request.args.get("task"))
    account_id = _positive_int("account_id")
    return jsonify({"task": task, "account_id": account_id, "cutoffs": _cutoffs(task, account_id)})


@app.get("/api/prediction")
def prediction():
    adapter = _adapter_by_identifier(request.args.get("adapter", ""))
    account_id = _positive_int("account_id")
    cutoff = _parse_cutoff(request.args.get("cutoff"))
    if cutoff.isoformat() not in _cutoffs(adapter.task, account_id):
        raise ValueError("cutoff is not valid for this account and task")
    snapshot = load_demo_snapshot(PROCESSED_DIR, adapter.task, account_id, cutoff)
    predictor = _predictor(adapter.identifier)
    start = _parse_optional_day(request.args.get("start"))
    history = snapshot.history_events.copy()
    if start is not None:
        if start.date() > snapshot.cutoff_day:
            raise ValueError("window start must be on or before the prediction cutoff")
        history = history.loc[pd.to_datetime(history["trans_date"]).dt.date.ge(start.date())].copy()
    if history.empty:
        raise ValueError("the selected window contains no transactions")
    result = predictor.predict_from_rows(
        profile_row=snapshot.prepared.profile_row,
        lifelong_events=snapshot.prepared.sample.profile_state.lifelong_events,
        transactions=history.to_dict(orient="records"),
        cutoff_time=cutoff,
    )
    return jsonify(
        _serialize_prediction(
            adapter,
            snapshot,
            predictor,
            result,
            visible_history=history,
            window_start=start.date().isoformat() if start is not None else None,
        )
    )


@app.post("/api/upload-prediction")
def upload_prediction():
    """Score one uploaded transaction history without retaining the file."""
    adapter = _adapter_by_identifier(request.form.get("adapter", ""))
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        raise ValueError("choose a CSV file before running a prediction")
    if not uploaded.filename.lower().endswith(".csv"):
        raise ValueError("the upload must be a .csv file")
    try:
        raw_text = uploaded.stream.read().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("the CSV must be UTF-8 encoded") from error
    try:
        events = pd.read_csv(StringIO(raw_text))
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError("the CSV could not be parsed") from error
    start = _parse_optional_day(request.form.get("start"))
    end = _parse_optional_day(request.form.get("end"))
    normalised, profile = _normalise_uploaded_events(events, start=start, end=end)
    cutoff = end or pd.Timestamp(normalised["trans_date"].max()).to_pydatetime()
    predictor = _predictor(adapter.identifier)
    result = predictor.predict_from_rows(
        profile_row=profile,
        lifelong_events=(),
        transactions=normalised.to_dict(orient="records"),
        cutoff_time=cutoff,
    )
    payload: dict[str, Any] = {
        "task": adapter.task,
        "adapter": {"id": adapter.identifier, "label": adapter.label, "rank": adapter.rank},
        "input": {
            "window_start": pd.Timestamp(normalised["trans_date"].min()).date().isoformat(),
            "cutoff": pd.Timestamp(cutoff).date().isoformat(),
            "event_count": len(normalised),
            "model_event_count": result.event_count,
            "context_limit": predictor.max_events,
            "context_truncated": result.context_truncated,
        },
    }
    if adapter.task == "future_value":
        payload["prediction"] = {
            "kind": "future_value",
            "prediction_log1p": _finite(result.predicted_log_future_volume),
            "prediction_volume": _finite(result.predicted_future_volume),
        }
    else:
        payload["prediction"] = {
            "kind": "cashflow_stress",
            "raw_logit": _finite(result.raw_model_output),
            "stress_probability": _finite(result.stress_probability),
        }
    return jsonify(payload)


@app.get("/api/training-history")
def training_history():
    metrics_path = _base_checkpoint().parent / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    return jsonify(
        {
            "epochs": [
                {
                    "epoch": int(row.epoch),
                    "train_loss": float(row.train_loss),
                    "validation_loss": float(row.validation_loss),
                }
                for row in metrics.itertuples(index=False)
            ],
            "best_validation_loss": float(metrics["validation_loss"].min()),
        }
    )


def _positive_int(name: str) -> int:
    try:
        value = int(request.args.get(name, ""))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_cutoff(value: str | None) -> datetime:
    if not value:
        raise ValueError("cutoff is required")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("cutoff must be an ISO date-time") from error


def _parse_optional_day(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("dates must use YYYY-MM-DD") from error
    if pd.isna(parsed):
        raise ValueError("dates must use YYYY-MM-DD")
    return parsed.to_pydatetime()


def _serialize_prediction(
    adapter: AdapterArtifact,
    snapshot: DemoSnapshot,
    predictor: AccountTaskPredictor,
    prediction: Any,
    *,
    visible_history: pd.DataFrame | None = None,
    window_start: str | None = None,
) -> dict[str, Any]:
    visible_ids = {
        transaction.transaction_id
        for transaction in snapshot.prepared.sample.transactions[-predictor.max_events :]
        if transaction.transaction_id is not None
    }
    typed_history = {transaction.transaction_id: transaction for transaction in snapshot.prepared.sample.transactions}
    raw_history = snapshot.history_events if visible_history is None else visible_history
    history = [
        _serialize_event(
            row._asdict(),
            transaction=typed_history.get(_optional_int(row.trans_id)),
            visible=_optional_int(row.trans_id) in visible_ids,
            predictor=predictor,
        )
        for row in raw_history.itertuples(index=False)
    ]
    future = [
        _serialize_event(row._asdict(), transaction=None, visible=False, predictor=None)
        for row in snapshot.future_events.itertuples(index=False)
    ]
    common: dict[str, Any] = {
        "adapter": {"id": adapter.identifier, "label": adapter.label, "rank": adapter.rank},
        "task": snapshot.task,
        "account": {"id": prediction.account_id, "split": snapshot.split},
        "cutoff": snapshot.cutoff_day.isoformat(),
        "window_end": snapshot.window_end_day.isoformat(),
        "history_event_count": len(raw_history),
        "model_event_count": prediction.event_count,
        "context_limit": predictor.max_events,
        "context_truncated": prediction.context_truncated,
        "window_start": window_start,
        "history": history,
        "future": future,
    }
    if snapshot.task == "future_value":
        actual_volume = snapshot.observed_future_volume or 0.0
        common["outcome"] = {
            "kind": "future_value",
            "prediction_log1p": _finite(prediction.predicted_log_future_volume),
            "prediction_volume": _finite(prediction.predicted_future_volume),
            "actual_log1p": _finite(float(snapshot.observed_target)),
            "actual_volume": _finite(actual_volume),
            "absolute_error_log1p": _finite(abs(prediction.raw_model_output - float(snapshot.observed_target))),
            "message": "Scalar regression on log1p of 180-day absolute transaction volume; it is not a token-logit or a calibrated range.",
        }
    else:
        common["outcome"] = {
            "kind": "cashflow_stress",
            "raw_logit": _finite(prediction.raw_model_output),
            "stress_probability": _finite(prediction.stress_probability),
            "actual_crossed_threshold": bool(snapshot.observed_target),
            "actual_minimum_balance": _finite(snapshot.observed_future_minimum_balance),
            "low_balance_threshold": _finite(snapshot.low_balance_threshold),
            "message": "Binary head: sigmoid(raw logit) gives the probability of crossing the train-fitted low-balance threshold in the next 60 days.",
        }
    return common


def _normalise_uploaded_events(
    events: pd.DataFrame,
    *,
    start: datetime | None,
    end: datetime | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the deliberately narrow custom-CSV inference contract.

    The model knows the Czech transaction schema.  Unknown categorical values
    are still accepted and map to the tokenizer's unknown token, but callers
    should keep the listed columns semantically aligned with their names.
    """
    required = {"trans_date", "amount", "balance", "trans_type"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    if len(events) > 10_000:
        raise ValueError("CSV has more than 10,000 rows; upload a smaller account window")
    if events.empty:
        raise ValueError("CSV has no transaction rows")
    normalised = events.copy()
    normalised["trans_date"] = pd.to_datetime(normalised["trans_date"], errors="coerce")
    normalised["amount"] = pd.to_numeric(normalised["amount"], errors="coerce")
    normalised["balance"] = pd.to_numeric(normalised["balance"], errors="coerce")
    if normalised[["trans_date", "amount", "balance"]].isna().any().any():
        raise ValueError("trans_date, amount, and balance must be complete and parseable")
    normalised["trans_type"] = normalised["trans_type"].astype("string").str.strip()
    if normalised["trans_type"].isna().any() or normalised["trans_type"].eq("").any():
        raise ValueError("trans_type must be present for every row")
    if "create_date" in normalised:
        created_values = pd.to_datetime(normalised["create_date"], errors="coerce").dropna()
        if created_values.empty:
            raise ValueError("create_date must be parseable when supplied")
        account_created = pd.Timestamp(created_values.min()).to_pydatetime()
    else:
        account_created = pd.Timestamp(normalised["trans_date"].min()).to_pydatetime()
    for column in ("operation", "category", "other_bank_id"):
        if column not in normalised:
            normalised[column] = None
    if "account_id" not in normalised:
        normalised["account_id"] = 1
    accounts = normalised["account_id"].dropna().unique()
    if len(accounts) != 1:
        raise ValueError("upload one account history per CSV")
    try:
        account_id = int(accounts[0])
    except (TypeError, ValueError) as error:
        raise ValueError("account_id must be an integer when supplied") from error
    normalised["account_id"] = account_id
    if "trans_id" not in normalised:
        normalised["trans_id"] = range(1, len(normalised) + 1)
    normalised["trans_id"] = pd.to_numeric(normalised["trans_id"], errors="coerce").fillna(0).astype(int)
    if start is not None:
        normalised = normalised.loc[normalised["trans_date"].dt.date.ge(start.date())]
    if end is not None:
        normalised = normalised.loc[normalised["trans_date"].dt.date.le(end.date())]
    normalised = normalised.sort_values(["trans_date", "trans_id"]).reset_index(drop=True)
    if normalised.empty:
        raise ValueError("no rows remain inside the requested start and end dates")
    profile = {
        "account_id": account_id,
        "create_date": account_created,
        "birth_date": None,
        "frequency": None,
        "gender": None,
        "region": None,
    }
    return normalised, profile


def _serialize_event(
    row: dict[str, Any],
    *,
    transaction: Any,
    visible: bool,
    predictor: AccountTaskPredictor | None,
) -> dict[str, Any]:
    item = {
        "id": _optional_int(row.get("trans_id")),
        "date": _iso_date(row.get("trans_date")),
        "amount": _finite(row.get("amount")),
        "balance": _finite(row.get("balance")),
        "type": _clean_text(row.get("trans_type")),
        "operation": _clean_text(row.get("operation")),
        "category": _clean_text(row.get("category")),
        "model_visible": visible,
        "amount_bucket": None,
        "balance_bucket": None,
    }
    if transaction is not None and predictor is not None:
        item["amount_bucket"] = transaction_bucket_label(predictor.tokenizers.event, transaction, "amount_abs")
        item["balance_bucket"] = transaction_bucket_label(predictor.tokenizers.event, transaction, "balance_after")
    return item


def _optional_int(value: Any) -> int | None:
    try:
        if _is_missing(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _iso_date(value: Any) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def _finite(value: Any) -> float | None:
    if _is_missing(value):
        return None
    number = float(value)
    return number if isfinite(number) else None


def _is_missing(value: Any) -> bool:
    """Recognise Python NaN and pandas.NA without asking pandas for a bool."""
    if value is None or type(value).__name__ == "NAType":
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
