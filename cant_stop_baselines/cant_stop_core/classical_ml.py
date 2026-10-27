"""Adapters and persistence for scikit-learn Stop/Continue policies."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

from .agents import Rule28Agent
from .model import Move, State
from .rl_env import StopContinueAction


MODEL_FORMAT_VERSION = 1
FEATURE_SIZES = {"raw": 58, "enhanced": 60, "domain": 68}


def _joblib() -> Any:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Classical ML support requires scikit-learn/joblib. Install "
            "requirements-ml.txt."
        ) from exc
    return joblib


def save_model_bundle(path: str | Path, *, model: Any, model_name: str,
                      feature_mode: str, prediction_target: str,
                      metadata: dict[str, Any] | None = None) -> None:
    if feature_mode not in FEATURE_SIZES:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
    if prediction_target not in ("action", "continue_margin"):
        raise ValueError("prediction_target must be action or continue_margin")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _joblib().dump({
        "format_version": MODEL_FORMAT_VERSION,
        "model": model,
        "model_name": model_name,
        "feature_mode": feature_mode,
        "feature_count": FEATURE_SIZES[feature_mode],
        "prediction_target": prediction_target,
        "metadata": metadata or {},
    }, destination)


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    bundle = _joblib().load(Path(path))
    if not isinstance(bundle, dict) or bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported classical ML model bundle")
    required = {"model", "model_name", "feature_mode", "feature_count",
                "prediction_target", "metadata"}
    if not required.issubset(bundle):
        raise ValueError("Incomplete classical ML model bundle")
    if bundle["feature_mode"] not in FEATURE_SIZES:
        raise ValueError("Model bundle has an unknown feature mode")
    if bundle["feature_count"] != FEATURE_SIZES[bundle["feature_mode"]]:
        raise ValueError("Model bundle feature count is inconsistent")
    return bundle


class ClassicalMLAgent:
    """Greedy sklearn Stop/Continue policy with fixed GA dice pairing."""

    def __init__(self, model: Any, *, feature_mode: str,
                 prediction_target: str = "action",
                 pair_heuristic: str = "ga", model_name: str = "classical-ml") -> None:
        if feature_mode not in FEATURE_SIZES:
            raise ValueError(f"Unknown feature mode: {feature_mode}")
        if prediction_target not in ("action", "continue_margin"):
            raise ValueError("prediction_target must be action or continue_margin")
        self.model = model
        self.feature_mode = feature_mode
        self.prediction_target = prediction_target
        self.model_name = model_name
        self.pairing_policy = Rule28Agent(pair_heuristic=pair_heuristic, smart=True)

    @classmethod
    def from_file(cls, path: str | Path) -> "ClassicalMLAgent":
        bundle = load_model_bundle(path)
        return cls(
            bundle["model"], feature_mode=bundle["feature_mode"],
            prediction_target=bundle["prediction_target"],
            model_name=bundle["model_name"],
        )

    def choose_move(self, state: State, moves: Sequence[Move],
                    rng: random.Random) -> Move:
        return self.pairing_policy.choose_move(state, moves, rng)

    def should_stop(self, state: State, rng: random.Random) -> bool:
        del rng
        features = state.features(
            enhanced=self.feature_mode in ("enhanced", "domain"),
            domain=self.feature_mode == "domain",
        )
        prediction = self.model.predict([features])[0]
        if self.prediction_target == "continue_margin":
            return float(prediction) <= 0.0
        return int(prediction) == int(StopContinueAction.STOP)
