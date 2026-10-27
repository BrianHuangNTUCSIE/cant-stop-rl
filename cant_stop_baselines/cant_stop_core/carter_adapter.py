"""Optional adapter for the public Carter Keras checkpoint.

The public artifact is loaded only when a caller supplies its local path.  The
adapter intentionally does not register it as a built-in agent: the original
project uses a different simulator and its 51-feature encoder has no exact
one-to-one representation in this engine.  Results should therefore be
reported as a checkpoint/feature adaptation unless the original environment
is reproduced separately.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

from .constants import COLUMNS, LENGTH
from .model import Move, State


class CarterAdapterError(RuntimeError):
    """Raised when the optional Carter adapter cannot be loaded or used."""


def carter_features(state: State) -> list[float]:
    """Build the verified 51-feature order from Carter's public source.

    The feature names and scaling follow ``rl_player.py``: own/opponent
    progress, current-turn marker progress, global completed-column mask,
    normalized roll count, Rule-of-28 score, success probability, active-marker
    count, and completed-column counts.  Rule and probability values are the
    corresponding values from this project, so this is an adaptation rather
    than a claim of exact original-environment replay.
    """
    player = state.current_player
    me = state.permanent[player]
    opponent = state.permanent[1 - player]
    active = state.active_dict()
    features = [me[i] / LENGTH[column] for i, column in enumerate(COLUMNS)]
    features += [opponent[i] / LENGTH[column] for i, column in enumerate(COLUMNS)]
    features += [
        (active[column] - me[column - 2]) / LENGTH[column]
        if column in active else 0.0
        for column in COLUMNS
    ]
    features += [float(owner != -1) for owner in state.owner]
    features.append(min(state.rolls_this_turn / 10.0, 1.0))

    from .rules import bust_probability, rule28_points
    features.append(rule28_points(state) / 50.0 if active else 0.0)
    features.append(1.0 - bust_probability(state) if len(active) == 3 else 1.0)
    features.append(len(active) / 3.0)
    features.append(state.completed(player) / 3.0)
    features.append(state.completed(1 - player) / 3.0)
    features.append(float(any(pos == LENGTH[column] for column, pos in active.items())))
    if len(features) != 51:
        raise CarterAdapterError(f"Carter feature encoder produced {len(features)} values")
    return features


class CarterKerasAgent:
    """Inference-only loader for Carter's public two-action Keras model."""

    def __init__(self, checkpoint: str | Path):
        try:
            from tensorflow import keras
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise CarterAdapterError(
                "CarterKerasAgent requires optional TensorFlow/Keras"
            ) from exc
        path = Path(checkpoint)
        if not path.is_file():
            raise CarterAdapterError(f"Carter checkpoint not found: {path}")
        try:
            self.model = keras.models.load_model(path, compile=False)
        except Exception as exc:  # pragma: no cover - depends on Keras runtime
            raise CarterAdapterError(f"Could not load Carter checkpoint {path}") from exc
        shape = getattr(self.model, "input_shape", None)
        if shape is not None and shape[-1] != 51:
            raise CarterAdapterError(f"Expected 51 model inputs, found {shape[-1]}")

    def choose_move(self, state: State, moves: Sequence[Move],
                    rng: random.Random) -> Move:
        if not moves:
            raise CarterAdapterError("Cannot choose a move from an empty list")
        # Carter's public policy uses a heuristic pair selector, not a learned
        # pairing action: prefer an existing temporary column, then proximity
        # to 7, preserving legal-move order for ties.
        active = state.active_dict()
        existing = [move for move in moves if any(c in active for c in move.columns)]
        candidates = existing or list(moves)
        return min(candidates, key=lambda move: (
            sum(abs(column - 7) for column in move.columns),
            move.columns,
        ))

    def should_stop(self, state: State, rng: random.Random) -> bool:
        values = self.model.predict([carter_features(state)], verbose=0)
        try:
            action = int(values[0].argmax())
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise CarterAdapterError("Carter model returned an invalid prediction") from exc
        # Public Carter code defines action 0 as stop and action 1 as continue.
        return action == 0
