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

from .constants import COLUMNS, LENGTH, MARK_VALUE, STEP_VALUE
from .model import Move, State


class CarterAdapterError(RuntimeError):
    """Raised when the optional Carter adapter cannot be loaded or used."""


from itertools import combinations

# Carter's CantStopBoard.Prob_Table, in lexicographic order for combinations
# of columns 2..12.  Keeping the published rounded values avoids silently
# substituting this project's exact dynamic probability calculation.
_CARTER_PROBABILITIES = (
    .52,.58,.68,.75,.76,.71,.63,.53,.44,.66,.76,.81,.82,.76,.74,
    .63,.55,.77,.81,.83,.76,.76,.71,.63,.86,.88,.83,.81,.76,.74,
    .89,.84,.83,.78,.78,.82,.82,.74,.74,.71,.64,.63,.58,.55,.44,
    .67,.74,.79,.8,.78,.76,.66,.58,.77,.79,.81,.78,.76,.71,.64,
    .86,.85,.83,.82,.76,.74,.89,.84,.84,.78,.78,.84,.83,.76,.76,
    .78,.71,.71,.66,.63,.53,.8,.85,.85,.8,.82,.78,.71,.89,.91,
    .86,.88,.83,.82,.9,.89,.88,.84,.83,.86,.88,.82,.81,.82,.76,
    .76,.76,.74,.63,.89,.9,.87,.86,.84,.82,.91,.85,.89,.84,.84,
    .87,.86,.83,.83,.8,.78,.76,.78,.76,.71,.92,.91,.9,.89,.89,
    .9,.91,.85,.88,.85,.81,.83,.8,.82,.76,.89,.89,.86,.86,.85,
    .79,.81,.79,.81,.75,.8,.77,.77,.74,.76,.68,.67,.66,.58,.52,
)
_CARTER_PROBABILITY = dict(zip(combinations(range(2, 13), 3), _CARTER_PROBABILITIES))


def _carter_probability(columns: tuple[int, ...]) -> float:
    return _CARTER_PROBABILITY.get(tuple(sorted(columns)), 0.0)


def carter_features(state: State, *, probability_mode: str = "table") -> list[float]:
    """Build the verified 51-feature order from Carter's public source.

    The feature names and scaling follow ``rl_player.py``: own/opponent
    progress, current-turn marker progress, global completed-column mask,
    normalized roll count, Rule-of-28 score, success probability, active-marker
    count, and completed-column counts.  ``probability_mode='table'`` uses
    Carter's static ``Prob_Table``; ``'dynamic'`` retains this project's exact
    rules calculation for an explicitly labeled adaptation.
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

    points = 0
    for column, position in active.items():
        steps = position - me[column - 2]
        points += (MARK_VALUE[column] if column in state.columns_started
                   else steps * STEP_VALUE[column])
        if column in state.columns_started:
            points += (steps - 1) * STEP_VALUE[column]
    if len(active) == 3:
        columns = tuple(active)
        if all(c % 2 for c in columns): points += 2
        elif all(c % 2 == 0 for c in columns): points -= 2
        if all(c < 8 for c in columns) or all(c > 6 for c in columns): points += 4
    features.append(points / 50.0 if active else 0.0)
    if len(active) == 3:
        if probability_mode == "table":
            probability = _carter_probability(tuple(active))
        elif probability_mode == "dynamic":
            from .rules import bust_probability
            probability = 1.0 - bust_probability(state)
        else:
            raise ValueError("probability_mode must be 'table' or 'dynamic'")
        features.append(probability)
    else:
        features.append(1.0)
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
        # Match Carter's released selector: return the first legal pairing
        # containing an active column; otherwise minimize distance to seven.
        active = state.active_dict()
        for move in moves:
            if any(column in active for column in move.columns):
                return move
        return min(moves, key=lambda move: sum(abs(column - 7) for column in move.columns))

    def should_stop(self, state: State, rng: random.Random) -> bool:
        values = self.model.predict([carter_features(state)], verbose=0)
        try:
            action = int(values[0].argmax())
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise CarterAdapterError("Carter model returned an invalid prediction") from exc
        # Public Carter code defines action 0 as stop and action 1 as continue.
        return action == 0


class CarterNumpyAgent:
    """Dependency-light inference equivalent for Carter's HDF5 weights.

    This is intentionally separate from :class:`CarterKerasAgent`: it makes
    the published checkpoint reproducible without TensorFlow while preserving
    the Dense/ReLU/Dense/ReLU/Dense network used by the public model.
    """

    def __init__(self, weights: str | Path):
        try:
            import h5py
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise CarterAdapterError(
                "CarterNumpyAgent requires optional h5py and numpy"
            ) from exc
        path = Path(weights)
        if not path.is_file():
            raise CarterAdapterError(f"Carter weights not found: {path}")
        with h5py.File(path, "r") as handle:
            self.weights = [
                (np.asarray(handle[f"layers/{layer}/vars/{index}"][:]),
                 np.asarray(handle[f"layers/{layer}/vars/{index + 1}"][:]))
                for layer in ("dense", "dense_1", "dense_2")
                for index in (0,)
            ]
        if tuple(self.weights[0][0].shape) != (51, 64):
            raise CarterAdapterError("Expected Carter 51 -> 64 input weights")

    def choose_move(self, state: State, moves: Sequence[Move],
                    rng: random.Random) -> Move:
        if not moves:
            raise CarterAdapterError("Cannot choose a move from an empty list")
        active = state.active_dict()
        for move in moves:
            if any(column in active for column in move.columns):
                return move
        return min(moves, key=lambda move: sum(abs(column - 7) for column in move.columns))

    def should_stop(self, state: State, rng: random.Random) -> bool:
        import numpy as np
        values = np.asarray(carter_features(state), dtype=np.float32)
        for weights, bias in self.weights[:-1]:
            values = np.maximum(values @ weights + bias, 0.0)
        weights, bias = self.weights[-1]
        return int(np.argmax(values @ weights + bias)) == 0
