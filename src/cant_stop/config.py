"""Validated runtime configuration shared by simulator backends."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .transition import TRANSITION_SCHEMA_VERSION


BACKENDS = ("python", "python-reference", "python-multiprocess")


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str = "python-reference"
    worker_count: int = 1
    base_seed: int = 0
    policy_version: int = 1
    transition_schema_version: int = TRANSITION_SCHEMA_VERSION
    native_commit: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
        if self.worker_count < 1:
            raise ValueError("worker_count must be positive")
        if self.policy_version < 1:
            raise ValueError("policy_version must be positive")
        if self.transition_schema_version != TRANSITION_SCHEMA_VERSION:
            raise ValueError("unsupported transition schema version")

    @property
    def canonical_backend(self) -> str:
        return "python-reference" if self.backend == "python" else self.backend

    def metadata(self) -> dict[str, object]:
        values = asdict(self)
        values["backend"] = self.canonical_backend
        values["seed_formula"] = "environment_seed = base_seed + episode_id"
        return values
