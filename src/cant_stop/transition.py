"""Versioned, backend-neutral transition records.

The schema deliberately contains only primitive numeric fields plus debugging
metadata.  Actor implementations may use any internal game representation as
long as they emit this contract to the learner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


TRANSITION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Transition:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool
    n_steps: int = 1
    counterfactual: tuple[float, float] = (0.0, 0.0)
    counterfactual_mask: bool = False
    episode_id: int = -1
    worker_id: int = -1
    step_id: int = -1

    @classmethod
    def create(
        cls,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float] | None,
        done: bool,
        *,
        observation_size: int,
        n_steps: int = 1,
        counterfactual: Sequence[float] = (0.0, 0.0),
        counterfactual_mask: bool = False,
        episode_id: int = -1,
        worker_id: int = -1,
        step_id: int = -1,
    ) -> "Transition":
        """Validate and normalize one transition.

        Terminal ``next_state=None`` values become a fixed-size zero vector so
        every backend has the same wire and replay representation.
        """
        if observation_size < 1:
            raise ValueError("observation_size must be positive")
        if len(state) != observation_size:
            raise ValueError("state has the wrong observation size")
        if next_state is not None and len(next_state) != observation_size:
            raise ValueError("next_state has the wrong observation size")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        if len(counterfactual) != 2:
            raise ValueError("counterfactual must contain Continue and Stop values")
        return cls(
            tuple(float(value) for value in state), int(action), float(reward),
            (tuple(0.0 for _ in range(observation_size)) if next_state is None
             else tuple(float(value) for value in next_state)),
            bool(done), int(n_steps),
            (float(counterfactual[0]), float(counterfactual[1])),
            bool(counterfactual_mask), int(episode_id), int(worker_id), int(step_id),
        )


@dataclass(frozen=True)
class TransitionBatch:
    """A validated collection suitable for one actor-to-learner message."""

    observation_size: int
    transitions: tuple[Transition, ...]
    schema_version: int = TRANSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRANSITION_SCHEMA_VERSION:
            raise ValueError("unsupported transition schema version")
        if self.observation_size < 1:
            raise ValueError("observation_size must be positive")
        if any(
            len(item.state) != self.observation_size
            or len(item.next_state) != self.observation_size
            for item in self.transitions
        ):
            raise ValueError("transition observation size does not match the batch")

    @classmethod
    def from_iterable(
        cls, observation_size: int, transitions: Iterable[Transition]
    ) -> "TransitionBatch":
        return cls(observation_size, tuple(transitions))

    def __len__(self) -> int:
        return len(self.transitions)

    def replay_fields(self) -> tuple[list, ...]:
        """Return columns in the order accepted by ReplayBuffer.push_batch."""
        items = self.transitions
        return (
            [item.state for item in items],
            [item.action for item in items],
            [item.reward for item in items],
            [item.next_state for item in items],
            [item.done for item in items],
            [item.n_steps for item in items],
            [item.counterfactual for item in items],
            [item.counterfactual_mask for item in items],
        )
