"""Train and test classical Stop/Continue models on a collected CSV dataset."""
from __future__ import annotations

import sys
from pathlib import Path

# Extend the import path to the project root so entry points work when
# launched as ``python scripts/<name>.py`` from the repository root.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
    import sklearn
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Classical ML training requires requirements-ml.txt"
    ) from exc

from cant_stop_core.classical_ml import FEATURE_SIZES, save_model_bundle


MODEL_NAMES = ("logistic", "decision-tree", "random-forest", "linear-svm", "ridge")
FEATURE_NAMES = tuple(FEATURE_SIZES)


def parse_names(value: str, choices: tuple[str, ...], label: str) -> tuple[str, ...]:
    if value == "all":
        return choices
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or any(name not in choices for name in names) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(
            f"{label} must be 'all' or unique comma-separated values from {', '.join(choices)}"
        )
    return names


def read_dataset(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Dataset is empty")
    feature_columns = [f"x{index:02d}" for index in range(68)]
    missing = set(feature_columns + ["game_id", "label", "continue_margin", "rollouts"]) - rows[0].keys()
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    features = np.asarray([
        [float(row[column]) for column in feature_columns] for row in rows
    ], dtype=np.float64)
    return rows, features


def grouped_split(rows: list[dict[str, str]], seed: int,
                  train_fraction: float = 0.70,
                  validation_fraction: float = 0.15) -> dict[str, np.ndarray]:
    """Split whole source games, never correlated states from the same game."""
    groups = sorted({row["game_id"] for row in rows})
    if len(groups) < 3:
        raise ValueError("At least three source games are required for grouped splits")
    random.Random(seed).shuffle(groups)
    train_count = max(1, int(len(groups) * train_fraction))
    validation_count = max(1, int(len(groups) * validation_fraction))
    if train_count + validation_count >= len(groups):
        train_count = len(groups) - 2
        validation_count = 1
    group_sets = {
        "train": set(groups[:train_count]),
        "validation": set(groups[train_count:train_count + validation_count]),
        "test": set(groups[train_count + validation_count:]),
    }
    return {
        name: np.asarray([
            index for index, row in enumerate(rows) if row["game_id"] in selected
        ], dtype=np.int64)
        for name, selected in group_sets.items()
    }


def candidate_models(name: str, seed: int) -> list[tuple[dict[str, Any], Any, str]]:
    """Return (reported parameters, estimator, target) candidates."""
    candidates: list[tuple[dict[str, Any], Any, str]] = []
    if name == "logistic":
        for c in (0.1, 1.0, 10.0):
            candidates.append((
                {"C": c},
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, max_iter=2_000, class_weight="balanced",
                                       random_state=seed),
                ),
                "action",
            ))
    elif name == "decision-tree":
        for depth in (3, 6, 12, None):
            for leaf in (5, 20):
                candidates.append((
                    {"max_depth": depth, "min_samples_leaf": leaf},
                    DecisionTreeClassifier(
                        max_depth=depth, min_samples_leaf=leaf,
                        class_weight="balanced", random_state=seed,
                    ),
                    "action",
                ))
    elif name == "random-forest":
        for depth in (6, 12, None):
            for leaf in (2, 10):
                candidates.append((
                    {"n_estimators": 200, "max_depth": depth,
                     "min_samples_leaf": leaf},
                    RandomForestClassifier(
                        n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
                        class_weight="balanced", random_state=seed, n_jobs=1,
                    ),
                    "action",
                ))
    elif name == "linear-svm":
        for c in (0.1, 1.0, 10.0):
            candidates.append((
                {"C": c},
                make_pipeline(
                    StandardScaler(),
                    LinearSVC(C=c, class_weight="balanced", random_state=seed,
                              max_iter=10_000),
                ),
                "action",
            ))
    elif name == "ridge":
        for alpha in (0.1, 1.0, 10.0, 100.0):
            candidates.append((
                {"alpha": alpha},
                make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
                "continue_margin",
            ))
    else:  # pragma: no cover - protected by CLI choices
        raise ValueError(f"Unknown model: {name}")
    return candidates


def fit_estimator(estimator: Any, target: str, x: np.ndarray,
                  labels: np.ndarray, margins: np.ndarray,
                  weights: np.ndarray) -> None:
    y = margins if target == "continue_margin" else labels
    if target == "continue_margin":
        estimator.fit(x, y, ridge__sample_weight=weights)
    elif hasattr(estimator, "steps"):
        final_name = estimator.steps[-1][0]
        estimator.fit(x, y, **{f"{final_name}__sample_weight": weights})
    else:
        estimator.fit(x, y, sample_weight=weights)


def prediction_scores(estimator: Any, target: str, x: np.ndarray) -> np.ndarray:
    if target == "continue_margin":
        return np.asarray(estimator.predict(x), dtype=np.float64)
    if hasattr(estimator, "predict_proba"):
        # Positive score means Continue (class 0), matching continue_margin.
        classes = list(estimator.classes_)
        return np.asarray(estimator.predict_proba(x)[:, classes.index(0)], dtype=np.float64)
    if hasattr(estimator, "decision_function"):
        # sklearn binary decision_function is positive for classes_[1] (= Stop).
        return -np.asarray(estimator.decision_function(x), dtype=np.float64)
    return -np.asarray(estimator.predict(x), dtype=np.float64)


def predict_actions(estimator: Any, target: str, x: np.ndarray) -> np.ndarray:
    if target == "continue_margin":
        return np.where(estimator.predict(x) > 0.0, 0, 1).astype(np.int64)
    return np.asarray(estimator.predict(x), dtype=np.int64)


def metrics(estimator: Any, target: str, x: np.ndarray,
            labels: np.ndarray, margins: np.ndarray) -> dict[str, Any]:
    predicted = predict_actions(estimator, target, x)
    wrong = predicted != labels
    scores = prediction_scores(estimator, target, x)
    result: dict[str, Any] = {
        "states": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision_stop": float(precision_score(labels, predicted, pos_label=1,
                                                  zero_division=0)),
        "recall_stop": float(recall_score(labels, predicted, pos_label=1,
                                           zero_division=0)),
        "f1_stop": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
        "mean_rollout_regret": float(np.mean(np.abs(margins) * wrong)),
        "confusion_matrix_rows_actual_continue_stop": confusion_matrix(
            labels, predicted, labels=[0, 1]
        ).tolist(),
        "actual_continue_rate": float(np.mean(labels == 0)),
        "predicted_continue_rate": float(np.mean(predicted == 0)),
    }
    try:
        result["roc_auc_continue"] = float(roc_auc_score(labels == 0, scores))
    except ValueError:
        result["roc_auc_continue"] = None
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--features", default="all")
    parser.add_argument("--seed", type=int, default=641)
    parser.add_argument("--min-margin", type=float, default=0.0,
                        help="Exclude ambiguous training rows only; validation/test stay intact")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/classical_ml/models"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        selected_models = parse_names(args.models, MODEL_NAMES, "models")
        selected_features = parse_names(args.features, FEATURE_NAMES, "features")
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.min_margin < 0.0:
        raise SystemExit("min-margin must be non-negative")
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    rows, all_features = read_dataset(args.dataset)
    splits = grouped_split(rows, args.seed)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    margins = np.asarray([float(row["continue_margin"]) for row in rows])
    rollout_counts = np.asarray([int(row["rollouts"]) for row in rows])
    # Preserve some weight for tied estimates while emphasizing costly mistakes.
    weights = np.maximum(np.abs(margins), 1.0 / rollout_counts)
    train_indices = splits["train"]
    if args.min_margin > 0.0:
        train_indices = train_indices[np.abs(margins[train_indices]) >= args.min_margin]
    if len(train_indices) == 0 or len(np.unique(labels[train_indices])) < 2:
        raise SystemExit("Filtered training split must contain both actions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "seed": args.seed,
        "min_training_margin": args.min_margin,
        "split_method": "70/15/15_grouped_by_source_game",
        "split_states": {name: int(len(index)) for name, index in splits.items()},
        "training_states_after_margin_filter": int(len(train_indices)),
        "runtime": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        "results": {},
    }

    for feature_mode in selected_features:
        feature_count = FEATURE_SIZES[feature_mode]
        x = all_features[:, :feature_count]
        for model_name in selected_models:
            key = f"{model_name}_{feature_mode}"
            best: tuple[float, float, dict[str, Any], Any, str, dict[str, Any]] | None = None
            for parameters, estimator, target in candidate_models(model_name, args.seed):
                fit_estimator(
                    estimator, target, x[train_indices], labels[train_indices],
                    margins[train_indices], weights[train_indices],
                )
                validation = metrics(
                    estimator, target, x[splits["validation"]],
                    labels[splits["validation"]], margins[splits["validation"]],
                )
                ranking = (
                    validation["mean_rollout_regret"],
                    -validation["balanced_accuracy"],
                )
                if best is None or ranking < (best[0], best[1]):
                    best = (ranking[0], ranking[1], parameters, estimator,
                            target, validation)
            assert best is not None
            _, _, parameters, estimator, target, validation = best
            # After validation-only hyperparameter selection, use all non-test
            # source games for the final artifact. The test split remains unseen.
            final_indices = np.concatenate((train_indices, splits["validation"]))
            if args.min_margin > 0.0:
                final_indices = final_indices[
                    np.abs(margins[final_indices]) >= args.min_margin
                ]
            estimator = clone(estimator)
            fit_estimator(
                estimator, target, x[final_indices], labels[final_indices],
                margins[final_indices], weights[final_indices],
            )
            test = metrics(
                estimator, target, x[splits["test"]], labels[splits["test"]],
                margins[splits["test"]],
            )
            artifact = args.output_dir / f"{key}.joblib"
            entry = {
                "model": model_name,
                "feature_mode": feature_mode,
                "prediction_target": target,
                "selected_parameters": parameters,
                "selection_metric": "minimum_validation_mean_rollout_regret_then_balanced_accuracy",
                "final_fit_states": int(len(final_indices)),
                "validation": validation,
                "test": test,
                "artifact": str(artifact),
            }
            save_model_bundle(
                artifact, model=estimator, model_name=model_name,
                feature_mode=feature_mode, prediction_target=target,
                metadata={
                    "dataset": str(args.dataset), "seed": args.seed,
                    "selected_parameters": parameters,
                    "validation_metrics": validation, "test_metrics": test,
                },
            )
            report["results"][key] = entry
            print(
                f"{key:<24} test_accuracy={test['accuracy']:.3f} "
                f"test_balanced={test['balanced_accuracy']:.3f} "
                f"test_regret={test['mean_rollout_regret']:.5f}"
            )

    report["elapsed_seconds"] = time.perf_counter() - started
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
