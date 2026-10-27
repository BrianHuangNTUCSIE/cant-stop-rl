"""Backend interfaces and factories for learner-facing environments."""
from __future__ import annotations

from typing import Protocol, Sequence

from .config import RuntimeConfig
from .model import State


class ActorEnvironmentState(Protocol):
    observation: list[float] | None
    info: dict[str, object]
    legal_action_mask: tuple[bool, ...]
    decision_state: State | None
    phase: str
    current_player: int


class EnvironmentActorBackend(Protocol):
    """Minimal batched interface used by a centralized learner."""

    worker_count: int

    def load_many(self, environments: Sequence[tuple[int, object]]) -> dict[int, ActorEnvironmentState]: ...
    def step_many(self, actions: Sequence[tuple[int, int]]) -> dict[int, object]: ...
    def load_basic_many(self, environments: Sequence[tuple[int, object]]) -> dict[int, list[float] | None]: ...
    def step_basic_many(self, actions: Sequence[tuple[int, int]]) -> dict[int, tuple[list[float] | None, float, bool, bool]]: ...
    def close(self) -> None: ...


def create_actor_backend(config: RuntimeConfig) -> EnvironmentActorBackend | None:
    """Create an actor backend; reference mode intentionally stays in-process."""
    if config.canonical_backend == "python-reference":
        return None
    if config.canonical_backend == "python-multiprocess":
        from rollout.actor_pool import ActorPool
        return ActorPool(config.worker_count)
    raise ValueError(f"unsupported backend: {config.backend}")
