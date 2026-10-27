"""Persistent multiprocessing actor pool with batched request dispatch."""
from __future__ import annotations

import atexit
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Sequence

from .actor import actor_main
from .protocol import ActorFailure, ActorState, StepResult


class ActorPoolError(RuntimeError):
    pass


class ActorPool:
    def __init__(self, worker_count: int, *, start_method: str = "spawn") -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        self.worker_count = worker_count
        context = mp.get_context(start_method)
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._closed = False
        for worker_id in range(worker_count):
            parent, child = context.Pipe()
            process = context.Process(
                target=actor_main, args=(worker_id, child),
                name=f"cant-stop-actor-{worker_id}", daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        atexit.register(self.close)

    def _validate_worker(self, worker_id: int) -> None:
        if self._closed:
            raise ActorPoolError("actor pool is closed")
        if worker_id < 0 or worker_id >= self.worker_count:
            raise ValueError("worker_id is outside the actor pool")
        if not self._processes[worker_id].is_alive():
            raise ActorPoolError(f"actor worker {worker_id} exited unexpectedly")

    def _request_many(self, requests: Sequence[tuple[int, str, object]]) -> dict[int, object]:
        worker_ids = [worker_id for worker_id, _, _ in requests]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("a batch may contain at most one request per worker")
        for worker_id, operation, payload in requests:
            self._validate_worker(worker_id)
            try:
                self._connections[worker_id].send((operation, payload))
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise ActorPoolError(
                    f"actor worker {worker_id} disconnected while sending {operation}"
                ) from exc
        results: dict[int, object] = {}
        for worker_id in worker_ids:
            try:
                status, payload = self._connections[worker_id].recv()
            except (EOFError, OSError) as exc:
                raise ActorPoolError(
                    f"actor worker {worker_id} disconnected"
                ) from exc
            if status == "error":
                assert isinstance(payload, ActorFailure)
                raise ActorPoolError(
                    f"worker {worker_id} failed during {payload.operation}: "
                    f"{payload.message}\n{payload.traceback}"
                )
            results[worker_id] = payload
        return results

    def load_many(self, environments: Sequence[tuple[int, object]]) -> dict[int, ActorState]:
        return self._request_many([
            (worker_id, "load", environment)
            for worker_id, environment in environments
        ])  # type: ignore[return-value]

    def load_basic_many(
        self, environments: Sequence[tuple[int, object]],
    ) -> dict[int, list[float] | None]:
        """Load environments without capturing board snapshots or metrics."""
        return self._request_many([
            (worker_id, "load_basic", environment)
            for worker_id, environment in environments
        ])  # type: ignore[return-value]

    def step_many(self, actions: Sequence[tuple[int, int]]) -> dict[int, StepResult]:
        return self._request_many([
            (worker_id, "step", action) for worker_id, action in actions
        ])  # type: ignore[return-value]

    def step_basic_many(
        self, actions: Sequence[tuple[int, int]],
    ) -> dict[int, tuple[list[float] | None, float, bool, bool]]:
        """Step without serializing state and info unused by simple learners."""
        return self._request_many([
            (worker_id, "step_basic", action) for worker_id, action in actions
        ])  # type: ignore[return-value]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for process, connection in zip(self._processes, self._connections):
            if process.is_alive():
                try:
                    connection.send(("shutdown", None))
                except (BrokenPipeError, OSError):
                    pass
        for process, connection in zip(self._processes, self._connections):
            if process.is_alive():
                try:
                    if connection.poll(2.0):
                        connection.recv()
                except (EOFError, OSError):
                    pass
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            connection.close()

    def __enter__(self) -> "ActorPool":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
