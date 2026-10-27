"""Reproducible Phase-0 simulator and actor-backend benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import json
import random
import time

from cant_stop_core.agents import make_agent
from cant_stop_core.game import Game
from cant_stop_core.rl_env import StopContinueEnv
from cant_stop_core.rules import legal_moves
from rollout.actor_pool import ActorPool


def timed(name: str, count: int, operation) -> dict[str, float | int | str]:
    started = time.perf_counter()
    checksum = operation()
    seconds = time.perf_counter() - started
    return {
        "name": name,
        "operations": count,
        "seconds": seconds,
        "operations_per_second": count / seconds if seconds else 0.0,
        "checksum": str(checksum),
    }


def representative_state(seed: int):
    game = Game()
    rng = random.Random(seed)
    while game.phase != "decision":
        moves = game.roll(rng)
        if moves:
            game.apply_move(moves[0])
    return game.snapshot()


def benchmark_primitives(iterations: int, seed: int, mc_rollouts: int) -> list[dict]:
    state = representative_state(seed)
    permanent = state.permanent[state.current_player]
    owner = state.owner
    active = state.active_dict()
    dice = (1, 2, 5, 6)
    rule28 = make_agent("rule28-smart")
    mc = make_agent("mc", mc_rollouts)
    rng = random.Random(seed)
    results = []
    results.append(timed("legal_moves", iterations, lambda: sum(
        len(legal_moves(dice, permanent, owner, active)) for _ in range(iterations)
    )))
    for feature_name, arguments in (
        ("features_raw", {}),
        ("features_enhanced", {"enhanced": True}),
        ("features_domain", {"domain": True}),
    ):
        results.append(timed(feature_name, iterations, lambda arguments=arguments: sum(
            len(state.features(**arguments)) for _ in range(iterations)
        )))
    results.append(timed("rule28_decision", iterations, lambda: sum(
        rule28.should_stop(state, rng) for _ in range(iterations)
    )))
    mc_iterations = max(1, iterations // 100)
    results.append(timed("mc_decision", mc_iterations, lambda: sum(
        mc.should_stop(state, rng) for _ in range(mc_iterations)
    )))
    return results


def make_env(seed: int) -> StopContinueEnv:
    return StopContinueEnv(
        opponent=make_agent("rule28-smart"), seed=seed, max_total_turns=2_000,
    )


def benchmark_environment(
    backend: str, episodes: int, workers: int, seed: int
) -> dict[str, float | int | str]:
    started = time.perf_counter()
    completed = transitions = checksum = total_turns = total_rolls = 0
    next_episode = 0
    worker_count = min(workers, episodes)

    if backend == "python-reference":
        environments = [make_env(seed + index) for index in range(worker_count)]
        states = [environment.reset()[0] for environment in environments]
        environment_steps = [0] * worker_count
        next_episode = worker_count
        while completed < episodes:
            for index, environment in enumerate(environments):
                if states[index] is None:
                    continue
                action = environment_steps[index] % 2
                observation, reward, terminated, truncated, info = environment.step(action)
                transitions += 1
                environment_steps[index] += 1
                states[index] = observation
                if terminated or truncated:
                    completed += 1
                    checksum += int(info.get("winner") or 0) + int(reward)
                    total_turns += sum(info["turns"])
                    total_rolls += sum(info["rolls"])
                    if next_episode < episodes:
                        environments[index] = make_env(seed + next_episode)
                        states[index] = environments[index].reset()[0]
                        environment_steps[index] = 0
                        next_episode += 1
        pool = None
    else:
        pool = ActorPool(worker_count)
        active = pool.load_many([
            (index, make_env(seed + index)) for index in range(worker_count)
        ])
        next_episode = worker_count
        worker_steps = [0] * worker_count
        while completed < episodes:
            results = pool.step_many([
                (worker_id, worker_steps[worker_id] % 2)
                for worker_id in sorted(active)
            ])
            reloads = []
            for worker_id, result in results.items():
                transitions += 1
                worker_steps[worker_id] += 1
                if result.terminated or result.truncated:
                    completed += 1
                    checksum += int(result.state.info.get("winner") or 0) + int(result.reward)
                    total_turns += sum(result.state.info["turns"])
                    total_rolls += sum(result.state.info["rolls"])
                    if next_episode < episodes:
                        reloads.append((worker_id, make_env(seed + next_episode)))
                        worker_steps[worker_id] = 0
                        next_episode += 1
                    else:
                        active.pop(worker_id)
                else:
                    active[worker_id] = result.state
            if reloads:
                active.update(pool.load_many(reloads))
        pool.close()

    seconds = time.perf_counter() - started
    return {
        "name": f"environment_{backend}",
        "episodes": episodes,
        "transitions": transitions,
        "workers": worker_count,
        "seconds": seconds,
        "episodes_per_second": episodes / seconds if seconds else 0.0,
        "transitions_per_second": transitions / seconds if seconds else 0.0,
        "turns_per_second": total_turns / seconds if seconds else 0.0,
        "rolls_per_second": total_rolls / seconds if seconds else 0.0,
        "checksum": f"{transitions}:{total_turns}:{total_rolls}:{checksum}",
    }


def benchmark_learner(iterations: int, seed: int) -> list[dict]:
    """Benchmark optional Torch work without making it a baseline dependency."""
    try:
        import torch
        from cant_stop_core.dqn import DQNConfig, QNetwork, ReplayBuffer, optimize_batch
    except ImportError:
        return [{"name": "torch", "available": False}]

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    config = DQNConfig(
        observation_size=68, action_size=2, hidden_sizes=(128, 128),
        batch_size=256, replay_capacity=1_024, algorithm="double-dqn",
    )
    online = QNetwork(68, 2, config.hidden_sizes)
    target = QNetwork(68, 2, config.hidden_sizes)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(1_024, 68, random.Random(seed))
    replay.push_batch(
        [[float((row + column) % 7) / 7.0 for column in range(68)]
         for row in range(1_024)],
        [row % 2 for row in range(1_024)],
        [0.0] * 1_024,
        [[float((row + column + 1) % 7) / 7.0 for column in range(68)]
         for row in range(1_024)],
        [False] * 1_024,
    )
    inference_iterations = max(1, iterations // 10)
    batch = torch.zeros((256, 68), dtype=torch.float32)
    def run_inference():
        with torch.no_grad():
            return sum(
                float(online(batch).sum().item())
                for _ in range(inference_iterations)
            )
    results = [timed(
        "network_inference_batch256", inference_iterations, run_inference
    )]
    update_iterations = max(1, iterations // 1_000)
    results.append(timed("optimize_batch", update_iterations, lambda: sum(
        float(optimize_batch(
            online, target, optimizer, replay, config, torch.device("cpu")
        ).item())
        for _ in range(update_iterations)
    )))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=441)
    parser.add_argument("--mc-rollouts", type=int, default=16)
    parser.add_argument(
        "--backend", choices=("python-reference", "python-multiprocess", "all"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.iterations, args.episodes, args.workers, args.mc_rollouts) < 1:
        parser.error("counts must be positive")

    report = {
        "seed": args.seed,
        "primitive_benchmarks": benchmark_primitives(
            args.iterations, args.seed, args.mc_rollouts
        ),
        "learner_benchmarks": benchmark_learner(args.iterations, args.seed),
        "backend_benchmarks": [],
    }
    backends = (
        ("python-reference", "python-multiprocess")
        if args.backend == "all" else (args.backend,)
    )
    for backend in backends:
        report["backend_benchmarks"].append(benchmark_environment(
            backend, args.episodes, args.workers, args.seed
        ))

    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
