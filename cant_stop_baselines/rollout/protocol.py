"""Pickle-safe actor process messages."""
from __future__ import annotations

from dataclasses import dataclass

from cant_stop_core.model import State


@dataclass(frozen=True)
class ActorState:
    observation: list[float] | None
    info: dict[str, object]
    legal_action_mask: tuple[bool, ...]
    decision_state: State | None
    phase: str
    current_player: int


@dataclass(frozen=True)
class StepResult:
    state: ActorState
    reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ActorFailure:
    worker_id: int
    operation: str
    message: str
    traceback: str


_MISSING = object()


def capture_state(env, observation=_MISSING, info=None) -> ActorState:
    active = env.game.winner is None and not env._truncated
    decision_state = env.game.snapshot() if active else None
    mask = tuple(env.legal_action_mask()) if active else tuple(
        False for _ in range(env.action_size)
    )
    if observation is _MISSING:
        observation = None if not active else env._observation()
    return ActorState(
        observation=observation,
        info=env._info() if info is None else info,
        legal_action_mask=mask,
        decision_state=decision_state,
        phase=env.game.phase,
        current_player=env.game.current_player,
    )
