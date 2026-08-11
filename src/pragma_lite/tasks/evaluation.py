"""Model-agnostic downstream metrics, baselines, and calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


def binary_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute ranking and probability metrics for a binary downstream task."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        roc_auc_score,
    )

    labels = _as_binary_labels(labels)
    probabilities = _as_probabilities(probabilities)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have matching shapes")
    if np.unique(labels).size != 2:
        raise ValueError("both target classes are required for binary metrics")
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "balanced_accuracy_at_threshold": float(
            balanced_accuracy_score(labels, probabilities >= threshold)
        ),
        "threshold": float(threshold),
        "positive_rate": float(labels.mean()),
        "n_examples": int(labels.size),
        "n_positive": int(labels.sum()),
    }


def prevalence_baseline_probabilities(train_labels: np.ndarray, size: int) -> np.ndarray:
    """Return the train-only constant-probability baseline for a split."""
    labels = _as_binary_labels(train_labels)
    if labels.size == 0:
        raise ValueError("train_labels must not be empty")
    return np.full(size, labels.mean(), dtype=np.float64)


def bootstrap_roc_auc_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    samples: int = 2_000,
    seed: int = 23,
) -> tuple[float, float]:
    """Bootstrap percentile interval for ROC-AUC on a fixed test set."""
    from sklearn.metrics import roc_auc_score

    labels = _as_binary_labels(labels)
    probabilities = _as_probabilities(probabilities)
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, labels.size, size=labels.size)
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size == 2:
            scores.append(float(roc_auc_score(sampled_labels, probabilities[indices])))
    if not scores:
        return float("nan"), float("nan")
    interval = np.quantile(scores, [0.025, 0.975])
    return float(interval[0]), float(interval[1])


@dataclass(frozen=True, slots=True)
class BinaryLogisticBenchmark:
    """Validation-selected logistic baseline and its locked-test report."""

    selected_c: float
    validation_average_precision_by_c: dict[float, float]
    validation_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    test_roc_auc_bootstrap_95_interval: tuple[float, float]
    estimator: Any

    def report(self) -> dict[str, Any]:
        """Return the serialisable result, excluding the fitted estimator."""
        return {
            "selected_logistic_c": self.selected_c,
            "validation_average_precision_by_c": self.validation_average_precision_by_c,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "test_roc_auc_bootstrap_95_interval": self.test_roc_auc_bootstrap_95_interval,
        }


def fit_binary_logistic_benchmark(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    valid_features: pd.DataFrame,
    valid_labels: np.ndarray,
    test_features: pd.DataFrame,
    test_labels: np.ndarray,
    *,
    c_values: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    class_weight: str | dict[int, float] | None = None,
    threshold: float = 0.5,
) -> BinaryLogisticBenchmark:
    """Tune regularisation on validation AP and evaluate test exactly once.

    ``class_weight`` defaults to ``None`` so model probabilities retain a
    chance of being calibrated.  Pass ``"balanced"`` only when ranking rare
    positives is the primary concern and report raw Brier scores carefully.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_labels = _as_binary_labels(train_labels)
    valid_labels = _as_binary_labels(valid_labels)
    test_labels = _as_binary_labels(test_labels)
    _validate_feature_splits(train_features, valid_features, test_features)
    candidate_values = tuple(float(value) for value in c_values)
    if not candidate_values or any(value <= 0.0 for value in candidate_values):
        raise ValueError("c_values must contain positive values")
    if np.unique(train_labels).size != 2 or np.unique(valid_labels).size != 2:
        raise ValueError("train and validation splits must both contain two classes")

    validation_ap: dict[float, float] = {}
    for c_value in candidate_values:
        candidate = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=2_000,
                random_state=17,
            ),
        )
        candidate.fit(train_features, train_labels)
        validation_ap[c_value] = float(
            average_precision_score(valid_labels, candidate.predict_proba(valid_features)[:, 1])
        )
    selected_c = max(validation_ap, key=validation_ap.get)
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=selected_c,
            class_weight=class_weight,
            max_iter=2_000,
            random_state=17,
        ),
    )
    estimator.fit(train_features, train_labels)
    validation_probabilities = estimator.predict_proba(valid_features)[:, 1]
    test_probabilities = estimator.predict_proba(test_features)[:, 1]
    return BinaryLogisticBenchmark(
        selected_c=selected_c,
        validation_average_precision_by_c=validation_ap,
        validation_metrics=binary_classification_metrics(
            valid_labels, validation_probabilities, threshold=threshold
        ),
        test_metrics=binary_classification_metrics(
            test_labels, test_probabilities, threshold=threshold
        ),
        test_roc_auc_bootstrap_95_interval=bootstrap_roc_auc_interval(
            test_labels, test_probabilities
        ),
        estimator=estimator,
    )


@dataclass(frozen=True, slots=True)
class PlattProbabilityCalibrator:
    """A validation-fitted logistic calibration map for held-out probabilities."""

    estimator: Any

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        logits = _probabilities_to_logits(probabilities).reshape(-1, 1)
        return self.estimator.predict_proba(logits)[:, 1]


def fit_platt_probability_calibrator(
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
) -> PlattProbabilityCalibrator:
    """Fit probability calibration only on validation labels, never test labels.

    With the current loan task's seven validation positives this is supplied as
    a framework capability, not a recommended default.  Use it once a task
    has enough validation positives to support a stable calibration curve.
    """
    from sklearn.linear_model import LogisticRegression

    labels = _as_binary_labels(validation_labels)
    if np.unique(labels).size != 2:
        raise ValueError("validation calibration requires both classes")
    estimator = LogisticRegression(C=1.0, max_iter=2_000, random_state=17)
    estimator.fit(_probabilities_to_logits(validation_probabilities).reshape(-1, 1), labels)
    return PlattProbabilityCalibrator(estimator=estimator)


@dataclass(frozen=True, slots=True)
class RidgeRegressionBenchmark:
    """Validation-selected future-value baseline and its locked-test report."""

    selected_alpha: float
    validation_mae_by_alpha: dict[float, float]
    validation_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    estimator: Any

    def report(self) -> dict[str, Any]:
        return {
            "selected_ridge_alpha": self.selected_alpha,
            "validation_mae_by_alpha": self.validation_mae_by_alpha,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
        }


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    """Evaluate a continuous future-value target."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("targets and predictions must be non-empty, matching vectors")
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "r2": float(r2_score(targets, predictions)),
        "n_examples": int(targets.size),
    }


def clustered_binary_bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    account_ids: np.ndarray,
    *,
    samples: int = 1_000,
    seed: int = 23,
) -> dict[str, tuple[float, float]]:
    """Bootstrap binary metrics by account, preserving repeated snapshots.

    Forward-looking task tables can contain several dated records per account.
    Resampling rows would overstate precision, so each draw resamples account
    clusters and takes every associated snapshot with it.
    """
    labels = _as_binary_labels(labels)
    probabilities = _as_probabilities(probabilities)
    indices_by_account = _cluster_indices(account_ids, labels.size)
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    values: dict[str, list[float]] = {"roc_auc": [], "average_precision": [], "brier_score": []}
    for indices in _iter_cluster_bootstrap_indices(indices_by_account, samples=samples, seed=seed):
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        sampled_probabilities = probabilities[indices]
        values["roc_auc"].append(float(roc_auc_score(sampled_labels, sampled_probabilities)))
        values["average_precision"].append(float(average_precision_score(sampled_labels, sampled_probabilities)))
        values["brier_score"].append(float(brier_score_loss(sampled_labels, sampled_probabilities)))
    return _percentile_intervals(values)


def clustered_regression_bootstrap_intervals(
    targets: np.ndarray,
    predictions: np.ndarray,
    account_ids: np.ndarray,
    *,
    samples: int = 1_000,
    seed: int = 23,
) -> dict[str, tuple[float, float]]:
    """Bootstrap MAE, RMSE, and R-squared by account cluster."""
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("targets and predictions must be non-empty, matching vectors")
    indices_by_account = _cluster_indices(account_ids, targets.size)
    values: dict[str, list[float]] = {"mae": [], "rmse": [], "r2": []}
    for indices in _iter_cluster_bootstrap_indices(indices_by_account, samples=samples, seed=seed):
        metrics = regression_metrics(targets[indices], predictions[indices])
        for name in values:
            values[name].append(float(metrics[name]))
    return _percentile_intervals(values)


def fit_ridge_regression_benchmark(
    train_features: pd.DataFrame,
    train_targets: np.ndarray,
    valid_features: pd.DataFrame,
    valid_targets: np.ndarray,
    test_features: pd.DataFrame,
    test_targets: np.ndarray,
    *,
    alpha_values: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> RidgeRegressionBenchmark:
    """Tune a standardised Ridge regression baseline using validation MAE."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _validate_feature_splits(train_features, valid_features, test_features)
    train_targets = np.asarray(train_targets, dtype=np.float64).reshape(-1)
    valid_targets = np.asarray(valid_targets, dtype=np.float64).reshape(-1)
    test_targets = np.asarray(test_targets, dtype=np.float64).reshape(-1)
    alpha_values = tuple(float(value) for value in alpha_values)
    if not alpha_values or any(value <= 0.0 for value in alpha_values):
        raise ValueError("alpha_values must contain positive values")

    validation_mae: dict[float, float] = {}
    for alpha in alpha_values:
        candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        candidate.fit(train_features, train_targets)
        validation_mae[alpha] = float(
            mean_absolute_error(valid_targets, candidate.predict(valid_features))
        )
    selected_alpha = min(validation_mae, key=validation_mae.get)
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=selected_alpha))
    estimator.fit(train_features, train_targets)
    return RidgeRegressionBenchmark(
        selected_alpha=selected_alpha,
        validation_mae_by_alpha=validation_mae,
        validation_metrics=regression_metrics(valid_targets, estimator.predict(valid_features)),
        test_metrics=regression_metrics(test_targets, estimator.predict(test_features)),
        estimator=estimator,
    )


def _validate_feature_splits(*feature_frames: pd.DataFrame) -> None:
    if not feature_frames or any(frame.empty for frame in feature_frames):
        raise ValueError("feature splits must not be empty")
    columns = tuple(feature_frames[0].columns)
    if any(tuple(frame.columns) != columns for frame in feature_frames[1:]):
        raise ValueError("all feature splits must have identical ordered columns")
    if any(frame.isna().any().any() for frame in feature_frames):
        raise ValueError("feature splits must not contain missing values")


def _as_binary_labels(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels).reshape(-1)
    if not np.isin(array, (0, 1)).all():
        raise ValueError("binary labels must contain only zero and one")
    return array.astype(np.int64, copy=False)


def _as_probabilities(probabilities: np.ndarray) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all() or (array < 0.0).any() or (array > 1.0).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    return array


def _cluster_indices(account_ids: np.ndarray, expected_size: int) -> tuple[np.ndarray, ...]:
    accounts = np.asarray(account_ids).reshape(-1)
    if accounts.size != expected_size or accounts.size == 0:
        raise ValueError("account_ids must be a non-empty vector aligned with predictions")
    unique_accounts, inverse = np.unique(accounts, return_inverse=True)
    if unique_accounts.size == 0:
        raise ValueError("at least one account is required")
    return tuple(np.flatnonzero(inverse == index) for index in range(unique_accounts.size))


def _iter_cluster_bootstrap_indices(
    clusters: tuple[np.ndarray, ...], *, samples: int, seed: int
) -> Iterable[np.ndarray]:
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        selected_clusters = rng.integers(0, len(clusters), size=len(clusters))
        yield np.concatenate([clusters[index] for index in selected_clusters])


def _percentile_intervals(values: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    return {
        name: (float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975)))
        if scores
        else (float("nan"), float("nan"))
        for name, scores in values.items()
    }


def _probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(_as_probabilities(probabilities), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))
