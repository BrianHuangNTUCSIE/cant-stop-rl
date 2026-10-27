"""Worker lifecycle for a single persistent environment actor."""
from __future__ import annotations

import traceback

from .protocol import ActorFailure, StepResult, capture_state


def actor_main(worker_id: int, connection) -> None:
    # Prevent nested PyTorch thread pools when an actor contains a DQN opponent.
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    env = None
    while True:
        try:
            operation, payload = connection.recv()
        except EOFError:
            return
        if operation == "shutdown":
            connection.send(("ok", None))
            return
        try:
            if operation == "load":
                env = payload
                observation, info = env.reset()
                result = capture_state(env, observation, info)
            elif operation == "load_basic":
                env = payload
                observation, _ = env.reset()
                result = observation
            elif operation == "step":
                if env is None:
                    raise RuntimeError("worker has no environment; load one first")
                observation, reward, terminated, truncated, info = env.step(payload)
                state = capture_state(env, observation, info)
                result = StepResult(state, reward, terminated, truncated)
            elif operation == "step_basic":
                if env is None:
                    raise RuntimeError("worker has no environment; load one first")
                observation, reward, terminated, truncated, _ = env.step(payload)
                result = (observation, reward, terminated, truncated)
            else:
                raise ValueError(f"unknown actor operation: {operation}")
            connection.send(("ok", result))
        except BaseException as exc:  # propagate worker failures to the learner
            connection.send(("error", ActorFailure(
                worker_id, operation, f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )))
