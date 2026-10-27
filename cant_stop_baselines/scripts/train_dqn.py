"""Train and evaluate a DQN/Double DQN Stop/Continue policy."""
from __future__ import annotations

import sys
from pathlib import Path

# Extend the import path to the project root so entry points work when
# launched as ``python scripts/<name>.py`` from the repository root.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import csv
import json
import os
import platform
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from cant_stop_core.agents import AGENT_NAMES, Agent, make_agent
from cant_stop_core.backend import create_actor_backend
from cant_stop_core.config import BACKENDS, RuntimeConfig
from cant_stop_core.evaluation import wilson_interval
from cant_stop_core.dqn import (
    DQNAgent,
    DQNConfig,
    NStepAccumulator,
    PrioritizedReplayBuffer,
    QNetwork,
    ReplayBuffer,
    available_device_types,
    choose_actions,
    epsilon_at_step,
    load_checkpoint,
    save_checkpoint,
    seed_torch,
    select_device,
)
from cant_stop_core.evaluation import evaluate
from cant_stop_core.rl_env import StopContinueEnv
from cant_stop_core.mc_distillation import RolloutConfig, label_state
from cant_stop_core.transition import Transition, TransitionBatch
from rollout.protocol import ActorState
from learners.dqn_learner import DQNLearner


def _counterfactual_label_task(
    state, config: RolloutConfig, seed: int,
) -> tuple[float, float]:
    """Process-pool entry point for an expensive MC counterfactual label."""
    return label_state(state, config=config, seed=seed)


@dataclass
class EpisodeSlot:
    episode: int
    env: StopContinueEnv | None
    observation: list[float] | None
    info: dict[str, object]
    opponent_name: str
    n_step: NStepAccumulator
    worker_id: int
    actor_state: ActorState | None
    steps: int = 0
    shaped_return: float = 0.0
    started_at: float = 0.0


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hidden sizes must be comma-separated integers"
        ) from exc
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be positive")
    return sizes


def format_duration(seconds: float) -> str:
    """Format a duration compactly for progress output."""
    if seconds < 0 or not (seconds < float("inf")):
        return "--"
    total_seconds = int(seconds + 0.5)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


OPPONENT_NAMES = AGENT_NAMES
EXTERNAL_OPPONENT_NAMES = ("dqn-opponent", "ridge-domain")


def _parse_weighted_opponents(
    value: str, *, normalize: bool,
) -> tuple[tuple[str, float], ...]:
    """Parse NAME:WEIGHT pairs, optionally normalizing their weights."""
    entries: list[tuple[str, float]] = []
    try:
        for item in value.split(","):
            name, raw_weight = item.strip().rsplit(":", 1)
            if name not in OPPONENT_NAMES + EXTERNAL_OPPONENT_NAMES:
                raise ValueError(f"unknown opponent {name!r}")
            weight = float(raw_weight)
            if weight <= 0:
                raise ValueError("weights must be positive")
            entries.append((name, weight))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "opponent mix must contain NAME:POSITIVE_WEIGHT entries; "
            f"names: {', '.join(OPPONENT_NAMES + EXTERNAL_OPPONENT_NAMES)}"
        ) from exc
    if not entries:
        raise argparse.ArgumentTypeError("opponent mix cannot be empty")
    if len({name for name, _ in entries}) != len(entries):
        raise argparse.ArgumentTypeError("opponent mix names must be unique")
    total = sum(weight for _, weight in entries)
    if normalize:
        return tuple((name, weight / total) for name, weight in entries)
    return tuple(entries)


def parse_opponent_mix(value: str) -> tuple[tuple[str, float], ...]:
    """Parse a complete opponent mix and normalize its weights."""
    return _parse_weighted_opponents(value, normalize=True)


def parse_self_play_anchor_mix(value: str) -> tuple[tuple[str, float], ...]:
    """Parse anchor weights without normalizing; remainder is self-play."""
    return _parse_weighted_opponents(value, normalize=False)


def parse_opponent_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("opponents cannot be empty")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("opponent names must be unique")
    return names


def mc_heavy_curriculum_mix(progress: float) -> tuple[tuple[str, float], ...]:
    """Return the controlled three-stage opponent schedule."""
    if progress < 0.25:
        return (
            ("rule28-smart", 0.45), ("mc", 0.30),
            ("rule28-ga-pure", 0.10), ("rule28", 0.10),
            ("random", 0.05),
        )
    if progress < 0.65:
        return (
            ("rule28-smart", 0.25), ("mc", 0.55),
            ("rule28-ga-pure", 0.10), ("rule28", 0.05),
            ("random", 0.05),
        )
    return (
        ("rule28-smart", 0.20), ("mc", 0.65),
        ("rule28-ga-pure", 0.10), ("random", 0.05),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dqn", "double-dqn"), default="double-dqn")
    parser.add_argument("--enhanced-features", action="store_true")
    parser.add_argument(
        "--domain-features", action="store_true",
        help="Use 68 features: raw + enhanced + decision-oriented aggregates",
    )
    parser.add_argument(
        "--strategic-features", action="store_true",
        help="Use 80 features: domain features plus per-column risk and threat",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"),
                        default="cpu",
                        help=("Training device (default: cpu; this small network is "
                              "usually slower on MPS due to dispatch overhead)"))
    parser.add_argument(
        "--torch-threads", type=int, default=None,
        help="CPU threads used by this training process (default: PyTorch default)",
    )
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--backend", choices=BACKENDS, default="python",
        help=("Environment backend: python/python-reference preserves the "
              "in-process trainer; python-multiprocess runs environment steps "
              "in persistent actor processes"),
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Actor processes for python-multiprocess (default: --num-envs)",
    )
    parser.add_argument("--seed", type=int, default=441)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: runs/<algorithm>_<features>_seed<seed>")
    parser.add_argument(
        "--hidden-sizes", type=parse_hidden_sizes, default=(256, 256, 256),
        help="Hidden layers (default: deep 256x256x256 network)",
    )
    parser.add_argument("--network-type", choices=("standard", "dueling"),
                        default="standard")
    parser.add_argument(
        "--pairing-policy", choices=("ga", "couto", "learned"), default="learned",
        help="Learner pairing policy; default jointly learns pairing and stop/continue",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=250_000)
    parser.add_argument(
        "--replay-type", choices=("uniform", "prioritized"), default="uniform",
        help="Replay sampling strategy (prioritized uses PER with IS correction)",
    )
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument(
        "--train-frequency", type=int, default=4,
        help="One gradient update per this many collected transitions",
    )
    parser.add_argument("--target-update-steps", type=int, default=5_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=300_000)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--n-step", type=int, default=1,
                        help="Discounted return horizon (recommended: 5)")
    parser.add_argument(
        "--reward-shaping", type=float, default=0.0, metavar="SCALE",
        help="Potential-based banked-progress shaping scale (recommended: 0.25)",
    )
    parser.add_argument(
        "--shaping-profile", choices=("progress", "race-aware"),
        default="progress",
        help="Potential used by reward shaping; default preserves the original potential",
    )
    parser.add_argument(
        "--opponent-mix", type=parse_opponent_mix,
        default=parse_opponent_mix(
            "rule28-smart:0.35,mc:0.35,rule28-ga-pure:0.1,rule28:0.1,random:0.1"
        ),
        help="Training distribution as NAME:WEIGHT comma-separated pairs",
    )
    parser.add_argument(
        "--training-mc-rollouts", type=int, default=16,
        help="MC rollouts used during training (default: 16; validation remains 64)",
    )
    parser.add_argument(
        "--training-mc-future-roll-limit", type=int, default=20,
        help="Maximum future rolls per training MC sample (default: 20)",
    )
    parser.add_argument(
        "--counterfactual-loss-weight", type=float, default=0.0,
        help="Auxiliary MC Stop/Continue loss weight; 0 disables it",
    )
    parser.add_argument(
        "--counterfactual-rollouts", type=int, default=8,
        help="MC rollouts per action for counterfactual labels",
    )
    parser.add_argument(
        "--counterfactual-every", type=int, default=8,
        help="Label every N learner decisions (default: 8)",
    )
    parser.add_argument(
        "--counterfactual-workers", type=int, default=0,
        help=("Persistent CPU workers for MC counterfactual labels; 0 disables "
              "parallel labeling, and a positive value enables it"),
    )
    parser.add_argument(
        "--counterfactual-target-scale", type=float, default=1.0,
        help="Scale applied to MC Continue-minus-Stop targets",
    )
    parser.add_argument(
        "--ridge-opponent", type=Path, default=None,
        help="Optional Ridge model used as ridge-domain in curriculum/validation",
    )
    parser.add_argument(
        "--dqn-opponent", type=Path, default=None,
        help="Optional pretrained DQN checkpoint used as a warm-up opponent",
    )
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=None,
        help="Initialize learner and optimizer from a prior checkpoint; episodes are additional fine-tuning episodes",
    )
    parser.add_argument(
        "--reference-replay", type=Path, default=None,
        help="Prefill replay with frozen reference transitions before fine-tuning",
    )
    parser.add_argument(
        "--reference-policy-weight", type=float, default=0.0,
        help="KL regularization weight keeping fine-tuning near the initial checkpoint",
    )
    parser.add_argument(
        "--reference-policy-temperature", type=float, default=1.0,
        help="Temperature for reference-policy KL regularization",
    )
    parser.add_argument(
        "--dqn-warmup-episodes", type=int, default=0,
        help="Use --dqn-opponent exclusively for this many initial episodes",
    )
    parser.add_argument(
        "--self-play", action="store_true",
        help="Train against frozen snapshots of the learner instead of the configured opponent mix",
    )
    parser.add_argument(
        "--self-play-snapshot-every", type=int, default=5_000,
        help="Add a frozen learner snapshot to the self-play pool every N episodes",
    )
    parser.add_argument(
        "--self-play-history-size", type=int, default=20,
        help="Maximum number of frozen self-play snapshots to retain",
    )
    parser.add_argument(
        "--self-play-latest-probability", type=float, default=0.5,
        help="Probability of selecting the newest self-play snapshot",
    )
    parser.add_argument(
        "--self-play-anchor-mix", type=parse_self_play_anchor_mix, default=(),
        help=("Fixed external anchors mixed into self-play, e.g. "
              "rule28-smart:0.10,mc:0.10; remaining probability is self-play"),
    )
    parser.add_argument(
        "--strong-curriculum", action="store_true",
        help="Progress from basic heuristics to MC, Couto, and ridge-domain",
    )
    parser.add_argument(
        "--mc-heavy-curriculum", action="store_true",
        help=("Progress from the baseline opponent mix to an MC-heavy mix; "
              "does not require an external Ridge model"),
    )
    parser.add_argument(
        "--validation-opponents", type=parse_opponent_names,
        default=parse_opponent_names("rule28-smart,mc"),
        help="Checkpoint selection opponents; score is their minimum win rate",
    )
    parser.add_argument(
        "--validation-mc-rollouts", type=int, default=64,
        help="MC rollouts used for validation and final evaluation (default: 64)",
    )
    parser.add_argument(
        "--validation-seed-blocks", type=int, default=1,
        help="Aggregate validation over this many disjoint deterministic seed blocks",
    )
    parser.add_argument(
        "--validation-workers", type=int, default=0,
        help=("Process workers for parallel validation blocks; 0 keeps validation "
              "in-process (default: 0)"),
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=("minimum-win-rate", "minimum-wilson-lower"),
        default="minimum-win-rate",
        help="Robust checkpoint selection objective",
    )
    parser.add_argument("--max-turns", type=int, default=2_000)
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5_000,
                        help="Run greedy validation every N completed episodes; default: 5000; 0 disables")
    parser.add_argument("--interim-eval-games", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10_000,
                        help="Save a snapshot every N completed episodes; default: 10000; 0 disables")
    parser.add_argument(
        "--collapse-bust-threshold", type=float, default=None,
        help=("Early-stop when aggregate greedy-validation busts/turn exceeds "
              "this threshold for --collapse-patience evaluations"),
    )
    parser.add_argument(
        "--collapse-patience", type=int, default=2,
        help="Consecutive collapsed validations required for early stop (default: 2)",
    )
    parser.add_argument(
        "--collapse-min-episodes", type=int, default=0,
        help="Do not count collapsed validations before this many episodes",
    )
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def cpu_agent(model: QNetwork, config: DQNConfig, *, enhanced_features: bool,
              domain_features: bool, strategic_features: bool) -> DQNAgent:
    copy = QNetwork(config.observation_size, config.action_size,
                    config.hidden_sizes, config.network_type)
    copy.load_state_dict({
        name: value.detach().cpu() for name, value in model.state_dict().items()
    })
    return DQNAgent(
        copy, enhanced_features=enhanced_features,
        domain_features=domain_features, strategic_features=strategic_features,
        pair_heuristic=config.pairing_policy, device=torch.device("cpu")
    )


def _validation_task(task: tuple) -> dict:
    """Run one independent validation block in a child process.

    The model and opponent are deliberately passed by value.  This prevents
    worker-local agent state from leaking between blocks and makes the result
    equivalent to the sequential path, whose game RNG streams are determined
    solely by ``seed``.
    """
    (model_state, config, enhanced_features, domain_features, strategic_features,
     opponent, games, seed, max_turns, include_records) = task
    # Avoid multiplying PyTorch's intra-op threads across validation workers.
    torch.set_num_threads(1)
    model = QNetwork(
        config.observation_size, config.action_size, config.hidden_sizes,
        config.network_type,
    )
    model.load_state_dict(model_state)
    learned = cpu_agent(
        model, config, enhanced_features=enhanced_features,
        domain_features=domain_features, strategic_features=strategic_features,
    )
    return evaluate(
        learned, opponent, games=games, seed=seed,
        max_total_turns=max_turns, include_game_records=include_records,
    )


def validation_reports(model: QNetwork, config: DQNConfig, *, games: int,
                       seed: int, max_turns: int,
                        enhanced_features: bool, domain_features: bool,
                       strategic_features: bool,
                       opponents: tuple[str, ...], mc_rollouts: int,
                       opponent_agents: dict[str, Agent] | None = None,
                       seed_blocks: int = 1,
                       include_records: bool = False,
                       validation_workers: int = 0) -> dict[str, dict]:
    if validation_workers < 0:
        raise ValueError("validation_workers must be non-negative")
    if seed_blocks < 1 or games % seed_blocks:
        raise ValueError("games must be divisible by validation seed blocks")
    games_per_block = games // seed_blocks
    if games_per_block % 2:
        raise ValueError("games per validation seed block must be even")
    reports: dict[str, dict] = {}
    tasks: list[tuple[str, Agent, int]] = []
    for name in opponents:
        opponent = ((opponent_agents or {}).get(name)
                    or make_agent(name, mc_rollouts))
        tasks.extend(
            (name, opponent, seed + block * 1_000_000)
            for block in range(seed_blocks)
        )

    if validation_workers > 0 and len(tasks) > 1:
        worker_count = min(validation_workers, len(tasks))
        model_state = {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }
        worker_tasks = [(
            model_state, config, enhanced_features, domain_features, strategic_features,
            task[1], games_per_block, task[2], max_turns, include_records,
        ) for task in tasks]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(_validation_task, worker_tasks))
    else:
        learned = cpu_agent(
            model, config, enhanced_features=enhanced_features,
            domain_features=domain_features, strategic_features=strategic_features,
        )
        results = [
            evaluate(
                learned, task[1], games=games_per_block, seed=task[2],
                max_total_turns=max_turns, include_game_records=include_records,
            )
            for task in tasks
        ]

    blocks_by_name: dict[str, list[dict]] = {name: [] for name in opponents}
    for (name, _opponent, _block_seed), result in zip(tasks, results):
        blocks_by_name[name].append(result)

    for name in opponents:
        blocks = blocks_by_name[name]
        if seed_blocks == 1:
            reports[name] = blocks[0]
            continue
        agents = []
        for index, agent_id in enumerate(("A", "B")):
            wins = sum(block["agents"][index]["wins"] for block in blocks)
            turns = sum(block["agents"][index]["total_turns"] for block in blocks)
            rolls = sum(block["agents"][index]["total_rolls"] for block in blocks)
            busts = sum(block["agents"][index]["total_busts"] for block in blocks)
            agents.append({
                "id": agent_id, "wins": wins, "win_rate": wins / games,
                "win_rate_wilson95": wilson_interval(wins, games),
                "seat0_wins": sum(block["agents"][index]["seat0_wins"] for block in blocks),
                "seat1_wins": sum(block["agents"][index]["seat1_wins"] for block in blocks),
                "total_turns": turns, "avg_turns_per_game": turns / games,
                "total_rolls": rolls, "avg_rolls_per_game": rolls / games,
                "total_busts": busts,
                "busts_per_turn": busts / turns if turns else 0.0,
                "total_stops": sum(block["agents"][index]["total_stops"] for block in blocks),
            })
        reports[name] = {
            "games": games, "seeds": [block["seed"] for block in blocks],
            "draws": sum(block["draws"] for block in blocks), "agents": agents,
        }
        if include_records:
            records = []
            for block_index, block in enumerate(blocks):
                records.extend({"seed_block": block_index, **record}
                               for record in block["game_records"])
            reports[name]["game_records"] = records
    return reports


def selection_score(reports: dict[str, dict],
                    metric: str = "minimum-win-rate") -> float:
    """Robust checkpoint objective: worst win rate over validation opponents."""
    if metric == "minimum-win-rate":
        return min(report["agents"][0]["win_rate"] for report in reports.values())
    if metric == "minimum-wilson-lower":
        return min(report["agents"][0]["win_rate_wilson95"][0]
                   for report in reports.values())
    raise ValueError(f"Unknown checkpoint metric: {metric}")


def validation_behavior(reports: dict[str, dict]) -> dict[str, float]:
    """Aggregate greedy-policy behavior over all validation opponents."""
    agents = [report["agents"][0] for report in reports.values()]
    turns = sum(agent["total_turns"] for agent in agents)
    rolls = sum(agent["total_rolls"] for agent in agents)
    busts = sum(agent["total_busts"] for agent in agents)
    stops = sum(agent["total_stops"] for agent in agents)
    decisions = rolls - busts
    return {
        "busts_per_turn": busts / turns if turns else 0.0,
        "stops_per_turn": stops / turns if turns else 0.0,
        "rolls_per_turn": rolls / turns if turns else 0.0,
        "greedy_stop_rate": stops / decisions if decisions else 0.0,
    }


def write_evaluation_bundle(path: Path, reports: dict[str, dict],
                            metric: str = "minimum-win-rate") -> None:
    summary = {
        "selection_metric": metric.replace("-", "_"),
        "selection_score": selection_score(reports, metric),
        "opponents": reports,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    positive = (args.episodes, args.num_envs, args.eval_games,
                args.interim_eval_games, args.log_every)
    if any(value < 1 for value in positive):
        raise SystemExit(
            "episodes, num-envs, evaluation game counts, and log-every must be positive"
        )
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise SystemExit("eval-every and checkpoint-every must be non-negative")
    if args.collapse_patience < 1:
        raise SystemExit("collapse-patience must be positive")
    if args.collapse_min_episodes < 0:
        raise SystemExit("collapse-min-episodes must be non-negative")
    if (args.collapse_bust_threshold is not None
            and not 0.0 <= args.collapse_bust_threshold <= 1.0):
        raise SystemExit("collapse-bust-threshold must be in [0, 1]")
    if args.collapse_bust_threshold is not None and args.eval_every == 0:
        raise SystemExit("collapse detection requires eval-every > 0")
    if args.n_step < 1:
        raise SystemExit("n-step must be positive")
    if args.torch_threads is not None and args.torch_threads < 1:
        raise SystemExit("torch-threads must be positive")
    if args.num_workers is not None and args.num_workers < 1:
        raise SystemExit("num-workers must be positive")
    if (args.training_mc_rollouts < 1 or args.training_mc_future_roll_limit < 1
            or args.validation_mc_rollouts < 1):
        raise SystemExit("MC rollout counts must be positive")
    if args.counterfactual_loss_weight < 0.0:
        raise SystemExit("counterfactual-loss-weight must be non-negative")
    if args.counterfactual_rollouts < 1 or args.counterfactual_every < 1:
        raise SystemExit("counterfactual rollout count and frequency must be positive")
    if args.counterfactual_workers < 0:
        raise SystemExit("counterfactual-workers must be non-negative")
    if args.counterfactual_target_scale <= 0.0:
        raise SystemExit("counterfactual-target-scale must be positive")
    if args.validation_seed_blocks < 1:
        raise SystemExit("validation-seed-blocks must be positive")
    if args.validation_workers < 0:
        raise SystemExit("validation-workers must be non-negative")
    if args.interim_eval_games % args.validation_seed_blocks:
        raise SystemExit("interim-eval-games must divide evenly into validation seed blocks")
    if (args.interim_eval_games // args.validation_seed_blocks) % 2:
        raise SystemExit("games in each validation seed block must be even")
    if args.reward_shaping < 0.0:
        raise SystemExit("reward-shaping must be non-negative")
    if args.ridge_opponent is not None and not args.ridge_opponent.is_file():
        raise SystemExit(f"Ridge opponent model not found: {args.ridge_opponent}")
    if args.dqn_opponent is not None and not args.dqn_opponent.is_file():
        raise SystemExit(f"DQN opponent checkpoint not found: {args.dqn_opponent}")
    if args.initial_checkpoint is not None and not args.initial_checkpoint.is_file():
        raise SystemExit(f"Initial checkpoint not found: {args.initial_checkpoint}")
    if args.dqn_warmup_episodes < 0:
        raise SystemExit("dqn-warmup-episodes must be non-negative")
    if args.dqn_warmup_episodes and args.dqn_opponent is None:
        raise SystemExit("--dqn-warmup-episodes requires --dqn-opponent")
    if args.self_play_snapshot_every < 1:
        raise SystemExit("self-play-snapshot-every must be positive")
    if args.self_play_history_size < 1:
        raise SystemExit("self-play-history-size must be positive")
    if not 0.0 <= args.self_play_latest_probability <= 1.0:
        raise SystemExit("self-play-latest-probability must be in [0, 1]")
    anchor_weight = sum(weight for _, weight in args.self_play_anchor_mix)
    if anchor_weight > 1.0 + 1e-9:
        raise SystemExit("self-play-anchor-mix weights must sum to at most 1")
    if args.self_play and args.dqn_warmup_episodes:
        raise SystemExit("--self-play and --dqn-warmup-episodes are mutually exclusive")
    if args.reference_policy_weight < 0.0:
        raise SystemExit("reference-policy-weight must be non-negative")
    if args.reference_policy_weight > 0.0 and args.initial_checkpoint is None:
        raise SystemExit("reference-policy-weight requires --initial-checkpoint")
    if args.reference_policy_temperature <= 0.0:
        raise SystemExit("reference-policy-temperature must be positive")
    if args.reference_replay is not None and not args.reference_replay.is_file():
        raise SystemExit(f"Reference replay not found: {args.reference_replay}")
    if args.strong_curriculum and args.ridge_opponent is None:
        raise SystemExit("--strong-curriculum requires --ridge-opponent")
    if args.strong_curriculum and args.mc_heavy_curriculum:
        raise SystemExit(
            "--strong-curriculum and --mc-heavy-curriculum are mutually exclusive"
        )

    external_opponents: dict[str, Agent] = {}
    if args.ridge_opponent is not None:
        from cant_stop_core.classical_ml import ClassicalMLAgent
        external_opponents["ridge-domain"] = ClassicalMLAgent.from_file(
            args.ridge_opponent
        )
    if args.dqn_opponent is not None:
        external_opponents["dqn-opponent"] = DQNAgent.from_checkpoint(
            args.dqn_opponent, device="cpu"
        )
    available_opponents = set(OPPONENT_NAMES) | set(external_opponents)
    unknown_training = set(name for name, _ in args.opponent_mix) - available_opponents
    if unknown_training:
        raise SystemExit(
            "Training opponent mix requires configured external opponents: "
            + ", ".join(sorted(unknown_training))
        )
    unknown_anchors = (
        set(name for name, _ in args.self_play_anchor_mix) - available_opponents
    )
    if unknown_anchors:
        raise SystemExit(
            "Self-play anchor mix requires configured opponents: "
            + ", ".join(sorted(unknown_anchors))
        )
    unknown_validation = set(args.validation_opponents) - available_opponents
    if unknown_validation:
        raise SystemExit(
            f"Unknown validation opponents: {', '.join(sorted(unknown_validation))}"
        )

    config = DQNConfig(
        observation_size=(
            (80 if args.strategic_features else
                           (68 if args.domain_features else
                           (60 if args.enhanced_features else 58)))
            + (18 if args.pairing_policy == "learned" else 0)
        ),
        action_size=(8 if args.pairing_policy == "learned" else 2),
        hidden_sizes=args.hidden_sizes,
        algorithm=args.algorithm,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        replay_type=args.replay_type,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        warmup_steps=args.warmup_steps,
        train_frequency=args.train_frequency,
        target_update_steps=args.target_update_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        gradient_clip=args.gradient_clip,
        n_step=args.n_step,
        network_type=args.network_type,
        pairing_policy=args.pairing_policy,
        counterfactual_loss_weight=args.counterfactual_loss_weight,
        counterfactual_target_scale=args.counterfactual_target_scale,
    )
    requested_workers = args.num_workers or args.num_envs
    runtime_config = RuntimeConfig(
        backend=args.backend,
        worker_count=(
            min(requested_workers, args.num_envs, args.episodes)
            if args.backend == "python-multiprocess" else 1
        ),
        base_seed=args.seed,
    )
    active_environment_count = min(args.num_envs, args.episodes)
    if runtime_config.canonical_backend == "python-multiprocess":
        active_environment_count = min(
            active_environment_count, runtime_config.worker_count
        )
    feature_name = ("strategic" if args.strategic_features else
                    ("domain" if args.domain_features else
                     ("enhanced" if args.enhanced_features else "raw")))
    architecture_name = "deep_joint" if (
        args.hidden_sizes == (256, 256, 256)
        and args.pairing_policy == "learned"
        and args.algorithm == "double-dqn"
    ) else "custom"
    output_dir = (args.output_dir if args.output_dir is not None else
                  Path("runs") / f"{args.algorithm}_{feature_name}_{architecture_name}_seed{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
    seed_torch(args.seed)
    exploration_rng = random.Random(args.seed ^ 0xD0C0FFEE)
    replay_rng = random.Random(args.seed ^ 0x5EEDBEEF)

    online = QNetwork(config.observation_size, config.action_size,
                      config.hidden_sizes, config.network_type).to(device)
    target = QNetwork(config.observation_size, config.action_size,
                      config.hidden_sizes, config.network_type).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    reference_network: QNetwork | None = None
    if config.replay_type == "prioritized":
        replay = PrioritizedReplayBuffer(
            config.replay_capacity, config.observation_size, replay_rng,
            alpha=config.per_alpha, beta=config.per_beta_start,
        )
    else:
        replay = ReplayBuffer(config.replay_capacity, config.observation_size, replay_rng)

    initial_checkpoint_episodes = 0
    initial_checkpoint_global_step = 0
    if args.initial_checkpoint is not None:
        initial_checkpoint = load_checkpoint(args.initial_checkpoint, device)
        checkpoint_config = dict(initial_checkpoint["config"])
        checkpoint_config["hidden_sizes"] = tuple(checkpoint_config["hidden_sizes"])
        allowed_finetune_changes = {
            "learning_rate", "epsilon_start", "epsilon_end", "epsilon_decay_steps",
            "warmup_steps",
        }
        comparable_checkpoint_config = {
            key: value for key, value in checkpoint_config.items()
            if key not in allowed_finetune_changes
        }
        comparable_config = {
            key: value for key, value in asdict(config).items()
            if key not in allowed_finetune_changes
        }
        if comparable_checkpoint_config != comparable_config:
            raise SystemExit(
                "initial checkpoint configuration does not match the requested "
                "training configuration"
            )
        online.load_state_dict(initial_checkpoint["online_state_dict"])
        target.load_state_dict(initial_checkpoint["target_state_dict"])
        optimizer.load_state_dict(initial_checkpoint["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = config.learning_rate
        if args.reference_policy_weight > 0.0:
            reference_network = QNetwork(
                config.observation_size, config.action_size,
                config.hidden_sizes, config.network_type,
            ).to(device)
            reference_network.load_state_dict(initial_checkpoint["online_state_dict"])
            reference_network.eval()
            for parameter in reference_network.parameters():
                parameter.requires_grad_(False)
        initial_checkpoint_episodes = int(initial_checkpoint.get("episodes", 0))
        initial_checkpoint_global_step = int(initial_checkpoint.get("global_step", 0))

    if args.reference_replay is not None:
        reference_data = torch.load(
            args.reference_replay, map_location="cpu", weights_only=False
        )
        required_fields = (
            "states", "actions", "rewards", "next_states", "dones",
            "n_steps", "counterfactuals", "counterfactual_masks",
        )
        if any(field not in reference_data for field in required_fields):
            raise SystemExit("reference replay is missing required transition fields")
        replay.push_batch(
            reference_data["states"], reference_data["actions"],
            reference_data["rewards"], reference_data["next_states"],
            reference_data["dones"], reference_data["n_steps"],
            reference_data["counterfactuals"],
            reference_data["counterfactual_masks"],
        )
        print(f"Prefilled replay with {len(replay):,} reference transitions")

    learner = DQNLearner(
        online, target, optimizer, replay, config, device,
        reference=reference_network,
        reference_weight=args.reference_policy_weight,
        reference_temperature=args.reference_policy_temperature,
    )

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "requested_device": args.device,
        "selected_device": str(device),
        "available_devices": available_device_types(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        **runtime_config.metadata(),
    }
    experiment = {
        "config": asdict(config),
        "episodes": args.episodes,
        "num_envs": args.num_envs,
        "active_environment_count": active_environment_count,
        **runtime_config.metadata(),
        "seed": args.seed,
        "max_turns": args.max_turns,
        "final_eval_games": args.eval_games,
        "eval_every": args.eval_every,
        "interim_eval_games": args.interim_eval_games,
        "checkpoint_every": args.checkpoint_every,
        "runtime": runtime,
        "reward": ("terminal_win_+1_loss_-1_plus_potential_shaping"
                   if args.reward_shaping else "terminal_only_win_+1_loss_-1"),
        "reward_shaping_scale": args.reward_shaping,
        "shaping_profile": args.shaping_profile,
        "return_estimator": f"{args.n_step}_step_discounted",
        "pairing_policy": args.pairing_policy,
        "opponent_mix": dict(args.opponent_mix),
        "initial_checkpoint": (
            str(args.initial_checkpoint)
            if args.initial_checkpoint is not None else None
        ),
        "initial_checkpoint_episodes": initial_checkpoint_episodes,
        "replay_buffer_initialized_from_scratch": args.initial_checkpoint is not None,
        "reference_replay": str(args.reference_replay) if args.reference_replay else None,
        "reference_policy_weight": args.reference_policy_weight,
        "reference_policy_temperature": args.reference_policy_temperature,
        "strong_curriculum": args.strong_curriculum,
        "mc_heavy_curriculum": args.mc_heavy_curriculum,
        "mc_heavy_curriculum_schedule": (
            [
                {
                    "fraction": [0.0, 0.25],
                    "opponent_mix": {
                        "rule28-smart": 0.45, "mc": 0.30,
                        "rule28-ga-pure": 0.10, "rule28": 0.10,
                        "random": 0.05,
                    },
                },
                {
                    "fraction": [0.25, 0.65],
                    "opponent_mix": {
                        "rule28-smart": 0.25, "mc": 0.55,
                        "rule28-ga-pure": 0.10, "rule28": 0.05,
                        "random": 0.05,
                    },
                },
                {
                    "fraction": [0.65, 1.0],
                    "opponent_mix": {
                        "rule28-smart": 0.20, "mc": 0.65,
                        "rule28-ga-pure": 0.10, "random": 0.05,
                    },
                },
            ] if args.mc_heavy_curriculum else None
        ),
        "ridge_opponent": (str(args.ridge_opponent)
                           if args.ridge_opponent is not None else None),
        "dqn_opponent": (str(args.dqn_opponent)
                          if args.dqn_opponent is not None else None),
        "dqn_warmup_episodes": args.dqn_warmup_episodes,
        "self_play": args.self_play,
        "self_play_snapshot_every": args.self_play_snapshot_every,
        "self_play_history_size": args.self_play_history_size,
        "self_play_latest_probability": args.self_play_latest_probability,
        "self_play_anchor_mix": dict(args.self_play_anchor_mix),
        "training_mc_rollouts": args.training_mc_rollouts,
        "training_mc_future_roll_limit": args.training_mc_future_roll_limit,
        "counterfactual_loss_weight": args.counterfactual_loss_weight,
        "counterfactual_rollouts": args.counterfactual_rollouts,
        "counterfactual_every": args.counterfactual_every,
        "counterfactual_target_scale": args.counterfactual_target_scale,
        "validation_opponents": args.validation_opponents,
        "validation_mc_rollouts": args.validation_mc_rollouts,
        "validation_seed_blocks": args.validation_seed_blocks,
        "validation_workers": args.validation_workers,
        "checkpoint_selection": args.checkpoint_metric,
        "collapse_guard": {
            "busts_per_turn_threshold": args.collapse_bust_threshold,
            "consecutive_validations": args.collapse_patience,
            "minimum_episodes": args.collapse_min_episodes,
        },
        "seat_schedule": "episode_index_mod_2",
        "validation_seed": args.seed + 10_000_000,
        "final_evaluation_seed": args.seed + 20_000_000,
    }
    (output_dir / "config.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    print(f"Device: {device} (available: {', '.join(available_device_types())})")
    print(
        f"Environment backend: {runtime_config.canonical_backend}; "
        f"active environments={active_environment_count}; "
        f"workers={runtime_config.worker_count}"
    )

    actor_backend = create_actor_backend(runtime_config)

    self_play_pool: list[tuple[str, Agent]] = []
    self_play_rng = random.Random(args.seed ^ 0x51E1F00D)

    def make_self_play_snapshot(episode: int) -> tuple[str, Agent]:
        """Create an immutable CPU policy snapshot for self-play."""
        snapshot = QNetwork(
            config.observation_size, config.action_size,
            config.hidden_sizes, config.network_type,
        )
        snapshot.load_state_dict({
            name: value.detach().cpu().clone()
            for name, value in online.state_dict().items()
        })
        snapshot.eval()
        return (
            f"self-play-snapshot-{episode}",
            DQNAgent(
                snapshot,
                enhanced_features=args.enhanced_features,
                domain_features=args.domain_features,
                strategic_features=args.strategic_features,
                pair_heuristic=args.pairing_policy,
                device=torch.device("cpu"),
            ),
        )

    if args.self_play:
        self_play_pool.append(make_self_play_snapshot(initial_checkpoint_episodes))

    def curriculum_mix(episode: int) -> tuple[tuple[str, float], ...]:
        if episode < args.dqn_warmup_episodes:
            return (("dqn-opponent", 1.0),)
        if args.mc_heavy_curriculum:
            progress = episode / max(args.episodes - 1, 1)
            return mc_heavy_curriculum_mix(progress)
        if not args.strong_curriculum:
            return args.opponent_mix
        progress = episode / max(args.episodes - 1, 1)
        if progress < 0.25:
            return parse_opponent_mix(
                "random:0.10,rule28:0.15,rule28-smart:0.45,mc:0.30"
            )
        if progress < 0.65:
            return (
                ("rule28-smart", 0.20), ("mc", 0.30),
                ("couto-ga-stop", 0.20), ("ridge-domain", 0.30),
            )
        return (
            ("mc", 0.35), ("couto-ga-stop", 0.15),
            ("couto", 0.15), ("ridge-domain", 0.35),
        )

    def opponent_for_episode(episode: int) -> tuple[str, Agent]:
        if args.self_play:
            anchor_mix = args.self_play_anchor_mix
            anchor_total = sum(weight for _, weight in anchor_mix)
            anchor_draw = self_play_rng.random()
            if anchor_draw < anchor_total:
                cumulative = 0.0
                selected = anchor_mix[-1][0]
                for name, weight in anchor_mix:
                    cumulative += weight
                    if anchor_draw < cumulative:
                        selected = name
                        break
                opponent = external_opponents.get(selected)
                if opponent is None:
                    opponent = make_agent(
                        selected, args.training_mc_rollouts,
                        args.training_mc_future_roll_limit,
                    )
                return selected, opponent
            # The newest snapshot is useful for learning current weaknesses;
            # historical snapshots prevent rapid forgetting and cycling.
            if (len(self_play_pool) == 1
                    or self_play_rng.random() < args.self_play_latest_probability):
                return self_play_pool[-1]
            return self_play_rng.choice(self_play_pool[:-1])
        # Depend only on seed and episode, not asynchronous completion order.
        rng = random.Random(args.seed ^ ((episode + 1) * 0x9E3779B1))
        draw = rng.random()
        cumulative = 0.0
        mix = curriculum_mix(episode)
        selected = mix[-1][0]
        for name, weight in mix:
            cumulative += weight
            if draw < cumulative:
                selected = name
                break
        opponent = external_opponents.get(selected)
        if opponent is None:
            opponent = make_agent(
                selected, args.training_mc_rollouts,
                args.training_mc_future_roll_limit,
            )
        return selected, opponent

    def new_slot(episode: int, worker_id: int) -> EpisodeSlot:
        if (
            args.log_every > 0
            and (episode % args.log_every == 0 or episode == args.episodes - 1)
        ):
            print(
                f"Starting episode {episode + 1}/{args.episodes} "
                "(initializing environment)...",
                flush=True,
            )
        started_at = time.perf_counter()
        opponent_name, opponent = opponent_for_episode(episode)
        if args.pairing_policy == "ga":
            pairing_policy = make_agent("rule28-smart")
        elif args.pairing_policy == "couto":
            pairing_policy = make_agent("couto")
        else:
            pairing_policy = None
        env = StopContinueEnv(
            enhanced_features=args.enhanced_features,
            domain_features=args.domain_features,
            strategic_features=args.strategic_features,
            learner_player=episode % 2,
            pairing_policy=pairing_policy,
            opponent=opponent,
            seed=args.seed + episode,
            max_total_turns=args.max_turns,
            reward_shaping=args.reward_shaping,
            shaping_gamma=args.gamma,
            shaping_profile=args.shaping_profile,
            learn_pairing=args.pairing_policy == "learned",
        )
        actor_state = None
        if actor_backend is None:
            observation, info = env.reset()
            local_env: StopContinueEnv | None = env
        else:
            actor_state = actor_backend.load_many(((worker_id, env),))[worker_id]
            observation, info = actor_state.observation, actor_state.info
            local_env = None
        return EpisodeSlot(
            episode, local_env, observation, info, opponent_name,
            NStepAccumulator(config.n_step, config.gamma),
            worker_id, actor_state,
            started_at=started_at,
        )

    def slot_legal_action_mask(slot: EpisodeSlot) -> list[bool] | tuple[bool, ...]:
        if slot.env is not None:
            return slot.env.legal_action_mask()
        assert slot.actor_state is not None
        return slot.actor_state.legal_action_mask

    def slot_decision_state(slot: EpisodeSlot):
        if slot.env is not None:
            return (
                slot.env.game.phase,
                slot.env.game.current_player,
                slot.env.learner_player,
                slot.env.game.snapshot(),
            )
        assert slot.actor_state is not None
        return (
            slot.actor_state.phase,
            slot.actor_state.current_player,
            slot.info["learner_player"],
            slot.actor_state.decision_state,
        )

    counterfactual_config = RolloutConfig(
        rollouts=args.counterfactual_rollouts,
        opponent_rollouts=max(4, args.counterfactual_rollouts // 2),
        future_roll_limit=args.training_mc_future_roll_limit,
        max_turns=args.max_turns,
    )
    counterfactual_executor = None
    if (args.counterfactual_loss_weight > 0.0
            and args.counterfactual_workers > 0):
        from concurrent.futures import ProcessPoolExecutor

        worker_count = min(
            args.counterfactual_workers,
            max(1, os.cpu_count() or 1),
        )
        counterfactual_executor = ProcessPoolExecutor(
            max_workers=worker_count,
        )
        print(
            f"Counterfactual MC workers: {worker_count} "
            "(single training seed; labels are computed in parallel)",
            flush=True,
        )

    episode_fields = [
        "episode", "env_seed", "learner_player", "opponent", "completion_order",
        "global_step", "gradient_updates", "episode_steps", "reward",
        "shaped_return",
        "winner", "truncated", "epsilon", "learner_turns", "learner_rolls",
        "learner_busts", "learner_stops",
    ]
    metric_fields = [
        "episodes", "global_step", "gradient_updates", "epsilon",
        "recent_win_rate", "rolling_1000_win_rate", "mean_td_loss",
        "stop_action_rate", "greedy_stop_rate", "mean_abs_q", "max_abs_q",
        "replay_size", "steps_per_second", "updates_per_second",
        "elapsed_seconds", "eta_seconds",
    ]
    evaluation_rows: list[dict[str, object]] = []
    recent_rewards: deque[float] = deque(maxlen=1_000)
    window_rewards: list[float] = []
    window_losses: list[torch.Tensor] = []
    window_actions = [0] * config.action_size
    global_step = initial_checkpoint_global_step
    # A checkpoint does not store the update counter separately. Reconstruct
    # it from global_step so fine-tuning does not replay all historical
    # gradient updates before processing its first new episode.
    gradient_updates = (
        max(0, (global_step - config.warmup_steps) // config.train_frequency)
        if args.initial_checkpoint is not None else 0
    )
    completed = 0
    # ``args.episodes`` is the number of additional fine-tuning episodes;
    # episode IDs in the new output run therefore start at zero.
    next_episode = 0
    next_log = args.log_every
    next_eval = args.eval_every if args.eval_every else None
    next_checkpoint = args.checkpoint_every if args.checkpoint_every else None
    next_self_play_snapshot = (
        args.self_play_snapshot_every if args.self_play else None
    )
    next_target_update = (
        (global_step // config.target_update_steps + 1)
        * config.target_update_steps
    )
    best_validation_rate = -1.0
    consecutive_collapsed_validations = 0
    early_stop_reason: str | None = None
    start_time = time.perf_counter()

    slots: list[EpisodeSlot] = []
    for worker_id in range(active_environment_count):
        slots.append(new_slot(next_episode, worker_id))
        next_episode += 1

    training_path = output_dir / "training_episodes.csv"
    metrics_path = output_dir / "training_metrics.csv"
    with training_path.open("w", newline="", encoding="utf-8") as episode_handle, \
            metrics_path.open("w", newline="", encoding="utf-8") as metric_handle:
        episode_writer = csv.DictWriter(episode_handle, fieldnames=episode_fields)
        metric_writer = csv.DictWriter(metric_handle, fieldnames=metric_fields)
        episode_writer.writeheader()
        metric_writer.writeheader()

        while completed < args.episodes:
            active = [slot for slot in slots if slot.observation is not None]
            if not active:
                # A game can reach the turn safety limit while reset() is
                # advancing through the opponent's first turn.  In that case
                # reset returns no learner observation, so there is no
                # transition to train on; count the episode as truncated and
                # replace it instead of treating the empty active set as a
                # bookkeeping failure.
                for slot in list(slots):
                    completed += 1
                    print(
                        f"Truncated episode {completed}/{args.episodes} "
                        "before the first learner observation; replacing it.",
                        flush=True,
                    )
                    recent_rewards.append(0.0)
                    window_rewards.append(0.0)
                    learner_player = slot.episode % 2
                    episode_writer.writerow({
                        "episode": slot.episode,
                        "env_seed": args.seed + slot.episode,
                        "learner_player": learner_player,
                        "opponent": slot.opponent_name,
                        "completion_order": completed,
                        "global_step": global_step,
                        "gradient_updates": gradient_updates,
                        "episode_steps": 0,
                        "reward": 0.0,
                        "shaped_return": 0.0,
                        "winner": slot.info.get("winner"),
                        "truncated": True,
                        "epsilon": epsilon_at_step(config, global_step),
                        "learner_turns": slot.info.get("turns", (None, None))[learner_player],
                        "learner_rolls": slot.info.get("rolls", (None, None))[learner_player],
                        "learner_busts": slot.info.get("busts", (None, None))[learner_player],
                        "learner_stops": slot.info.get("stops", (None, None))[learner_player],
                    })
                    if completed < args.episodes:
                        slot_index = next(
                            index for index, current in enumerate(slots)
                            if current is slot
                        )
                        slots[slot_index] = new_slot(next_episode, slot.worker_id)
                        next_episode += 1
                    else:
                        slot_index = next(
                            index for index, current in enumerate(slots)
                            if current is slot
                        )
                        slots.pop(slot_index)
                episode_handle.flush()
                continue
            observations = [slot.observation for slot in active]
            epsilon = epsilon_at_step(config, global_step)
            actions = choose_actions(
                online, observations, epsilon, exploration_rng, device,
                [slot_legal_action_mask(slot) for slot in active]
                if args.pairing_policy == "learned" else None,
            )

            counterfactual_labels: list[tuple[object, bool]] = []
            counterfactual_futures = []
            for slot in active:
                phase, current_player, learner_player, decision_state = (
                    slot_decision_state(slot)
                )
                counterfactual = None
                counterfactual_mask = False
                if (args.counterfactual_loss_weight > 0.0
                        and phase == "decision"
                        and current_player == learner_player
                        and slot.steps % args.counterfactual_every == 0):
                    assert decision_state is not None
                    label_seed = (
                        args.seed ^ ((slot.episode + 1) * 0x9E3779B1)
                        ^ ((slot.steps + 1) * 0x85EBCA6B)
                    )
                    if counterfactual_executor is None:
                        counterfactual = label_state(
                            decision_state, config=counterfactual_config,
                            seed=label_seed,
                        )
                    else:
                        counterfactual_futures.append(
                            counterfactual_executor.submit(
                                _counterfactual_label_task,
                                decision_state, counterfactual_config, label_seed,
                            )
                        )
                    counterfactual_mask = True
                counterfactual_labels.append((counterfactual, counterfactual_mask))

            remote_results = (
                actor_backend.step_many([
                    (slot.worker_id, action)
                    for slot, action in zip(active, actions)
                ])
                if actor_backend is not None else {}
            )

            if counterfactual_futures:
                future_index = 0
                resolved_labels: list[tuple[object, bool]] = []
                for counterfactual, masked in counterfactual_labels:
                    if masked and counterfactual is None:
                        counterfactual = counterfactual_futures[future_index].result()
                        future_index += 1
                    resolved_labels.append((counterfactual, masked))
                counterfactual_labels = resolved_labels

            batch_transitions: list[Transition] = []
            finished: list[tuple[EpisodeSlot, float, bool, dict[str, object]]] = []
            for slot, action, (counterfactual, counterfactual_mask) in zip(
                active, actions, counterfactual_labels
            ):
                state = slot.observation
                assert state is not None
                if slot.env is not None:
                    next_observation, reward, terminated, truncated, info = (
                        slot.env.step(action)
                    )
                else:
                    result = remote_results[slot.worker_id]
                    slot.actor_state = result.state
                    next_observation = result.state.observation
                    reward = result.reward
                    terminated = result.terminated
                    truncated = result.truncated
                    info = result.state.info
                slot.shaped_return += reward
                done = terminated or truncated
                for transition in slot.n_step.append(
                    state, action, reward, next_observation, done,
                    counterfactual, counterfactual_mask,
                ):
                    batch_transitions.append(Transition.create(
                        transition.state, transition.action, transition.reward,
                        transition.next_state, transition.done,
                        observation_size=config.observation_size,
                        n_steps=transition.n_steps,
                        counterfactual=transition.counterfactual,
                        counterfactual_mask=transition.counterfactual_mask,
                        episode_id=slot.episode,
                        worker_id=slot.worker_id,
                        step_id=slot.steps,
                    ))
                window_actions[action] += 1
                slot.observation = next_observation
                slot.info = info
                slot.steps += 1
                if done:
                    finished.append((slot, reward, truncated, info))

            replay.push_transition_batch(
                TransitionBatch.from_iterable(
                    config.observation_size, batch_transitions
                )
            )
            global_step += len(active)

            gradient_updates, losses = learner.optimize_due(
                global_step, gradient_updates
            )
            window_losses.extend(losses)

            while global_step >= next_target_update:
                learner.update_target()
                next_target_update += config.target_update_steps

            for slot, reward, truncated, info in finished:
                learner_player = slot.episode % 2
                completed += 1
                episode_seconds = time.perf_counter() - slot.started_at
                elapsed = time.perf_counter() - start_time
                average_seconds = elapsed / completed
                remaining_seconds = average_seconds * (args.episodes - completed)
                if (
                    args.log_every > 0
                    and (completed % args.log_every == 0 or completed == args.episodes)
                ):
                    print(
                        f"Completed episode {completed}/{args.episodes} "
                        f"in {format_duration(episode_seconds)}; "
                        f"average={format_duration(average_seconds)}; "
                        f"ETA={format_duration(remaining_seconds)}",
                        flush=True,
                    )
                extrinsic_reward = float(info.get("extrinsic_reward", reward))
                recent_rewards.append(extrinsic_reward)
                window_rewards.append(extrinsic_reward)
                episode_writer.writerow({
                    "episode": slot.episode,
                    "env_seed": args.seed + slot.episode,
                    "learner_player": learner_player,
                    "opponent": slot.opponent_name,
                    "completion_order": completed,
                    "global_step": global_step,
                    "gradient_updates": gradient_updates,
                    "episode_steps": slot.steps,
                    "reward": extrinsic_reward,
                    "shaped_return": slot.shaped_return,
                    "winner": info.get("winner"),
                    "truncated": truncated,
                    "epsilon": epsilon_at_step(config, global_step),
                    "learner_turns": info.get("turns", (None, None))[learner_player],
                    "learner_rolls": info.get("rolls", (None, None))[learner_player],
                    "learner_busts": info.get("busts", (None, None))[learner_player],
                    "learner_stops": info.get("stops", (None, None))[learner_player],
                })
                if next_episode < args.episodes:
                    # EpisodeSlot is a dataclass, so list.index() uses value
                    # equality.  Slot values can become equal after an
                    # environment finishes, which can replace/remove the
                    # wrong slot and silently lose one episode.  These are
                    # live objects; locate the exact object instead.
                    slot_index = next(
                        index for index, current in enumerate(slots)
                        if current is slot
                    )
                    slots[slot_index] = new_slot(next_episode, slot.worker_id)
                    next_episode += 1
                else:
                    slot_index = next(
                        index for index, current in enumerate(slots)
                        if current is slot
                    )
                    slots.pop(slot_index)

            if completed >= next_log:
                diagnostic_observations = [
                    slot.observation for slot in slots
                    if slot.observation is not None
                ]
                diagnostic_q = None
                if diagnostic_observations:
                    with torch.no_grad():
                        diagnostic_q = online(torch.tensor(
                            diagnostic_observations, dtype=torch.float32,
                            device=device,
                        ))
                synchronize(device)
                elapsed = time.perf_counter() - start_time
                episode_rate = completed / elapsed if elapsed > 0 else 0.0
                eta_seconds = (
                    max(args.episodes - completed, 0) / episode_rate
                    if episode_rate > 0 else float("inf")
                )
                mean_loss = (float(torch.stack(window_losses).mean().cpu().item())
                             if window_losses else "")
                action_total = sum(window_actions)
                if diagnostic_q is None:
                    greedy_stop_rate = mean_abs_q = max_abs_q = ""
                else:
                    stop_action = 7 if args.pairing_policy == "learned" else 1
                    greedy_stop_rate = float(
                        (diagnostic_q.argmax(dim=1) == stop_action).float().mean().cpu().item()
                    )
                    mean_abs_q = float(diagnostic_q.abs().mean().cpu().item())
                    max_abs_q = float(diagnostic_q.abs().max().cpu().item())
                metric_writer.writerow({
                    "episodes": completed,
                    "global_step": global_step,
                    "gradient_updates": gradient_updates,
                    "epsilon": epsilon_at_step(config, global_step),
                    "recent_win_rate": (sum(x == 1.0 for x in window_rewards)
                                        / len(window_rewards) if window_rewards else 0.0),
                    "rolling_1000_win_rate": (
                        sum(x == 1.0 for x in recent_rewards) / len(recent_rewards)
                    ),
                    "mean_td_loss": mean_loss,
                    "stop_action_rate": (window_actions[7 if args.pairing_policy == "learned" else 1] / action_total
                                          if action_total else 0.0),
                    "greedy_stop_rate": greedy_stop_rate,
                    "mean_abs_q": mean_abs_q,
                    "max_abs_q": max_abs_q,
                    "replay_size": len(replay),
                    "steps_per_second": global_step / elapsed,
                    "updates_per_second": gradient_updates / elapsed,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": (eta_seconds if eta_seconds != float("inf")
                                    else ""),
                })
                print(
                    f"Episodes {completed}/{args.episodes}  steps={global_step}  "
                    f"updates={gradient_updates}  "
                    f"epsilon={epsilon_at_step(config, global_step):.3f}  "
                    f"recent_win={(sum(x == 1.0 for x in window_rewards) / len(window_rewards) if window_rewards else 0.0):.1%}  "
                    f"rolling1000={sum(x == 1.0 for x in recent_rewards) / len(recent_rewards):.1%}  "
                    f"steps/s={global_step / elapsed:.1f}  "
                    f"ETA={format_duration(eta_seconds)}"
                )
                episode_handle.flush()
                metric_handle.flush()
                window_rewards.clear()
                window_losses.clear()
                window_actions[:] = [0] * config.action_size
                while next_log <= completed:
                    next_log += args.log_every

            if next_checkpoint is not None and completed >= next_checkpoint:
                save_checkpoint(
                    output_dir / f"checkpoint_episode_{completed}.pt",
                    online, target, optimizer, config,
                    global_step=global_step, episodes=completed, metadata=runtime,
                )
                while next_checkpoint <= completed:
                    next_checkpoint += args.checkpoint_every

            if (next_self_play_snapshot is not None
                    and completed >= next_self_play_snapshot):
                snapshot_episode = next_self_play_snapshot
                snapshot_name, snapshot_agent = make_self_play_snapshot(
                    snapshot_episode
                )
                self_play_pool.append((snapshot_name, snapshot_agent))
                if len(self_play_pool) > args.self_play_history_size:
                    del self_play_pool[:-args.self_play_history_size]
                print(
                    f"Added {snapshot_name}; self-play pool size={len(self_play_pool)}"
                )
                while next_self_play_snapshot <= completed:
                    next_self_play_snapshot += args.self_play_snapshot_every

            if next_eval is not None and completed >= next_eval:
                reports = validation_reports(
                    online, config, games=args.interim_eval_games,
                    seed=args.seed + 10_000_000, max_turns=args.max_turns,
                    enhanced_features=args.enhanced_features,
                    domain_features=args.domain_features,
                    strategic_features=args.strategic_features,
                    opponents=args.validation_opponents,
                    mc_rollouts=args.validation_mc_rollouts,
                    opponent_agents=external_opponents,
                    seed_blocks=args.validation_seed_blocks,
                    validation_workers=args.validation_workers,
                )
                score = selection_score(reports, args.checkpoint_metric)
                behavior = validation_behavior(reports)
                mean_rate = sum(
                    report["agents"][0]["win_rate"]
                    for report in reports.values()
                ) / len(reports)
                for opponent_name, report in reports.items():
                    evaluation_rows.append({
                        "episodes": completed,
                        "global_step": global_step,
                        "opponent": opponent_name,
                        "games": args.interim_eval_games,
                        "wins": report["agents"][0]["wins"],
                        "win_rate": report["agents"][0]["win_rate"],
                        "draws": report["draws"],
                        "busts_per_turn": report["agents"][0]["busts_per_turn"],
                        "stops_per_turn": (
                            report["agents"][0]["total_stops"]
                            / report["agents"][0]["total_turns"]
                            if report["agents"][0]["total_turns"] else 0.0
                        ),
                        "rolls_per_turn": (
                            report["agents"][0]["total_rolls"]
                            / report["agents"][0]["total_turns"]
                            if report["agents"][0]["total_turns"] else 0.0
                        ),
                        "greedy_stop_rate": (
                            report["agents"][0]["total_stops"]
                            / (report["agents"][0]["total_rolls"]
                               - report["agents"][0]["total_busts"])
                            if (report["agents"][0]["total_rolls"]
                                - report["agents"][0]["total_busts"]) else 0.0
                        ),
                        "aggregate_busts_per_turn": behavior["busts_per_turn"],
                        "aggregate_stops_per_turn": behavior["stops_per_turn"],
                        "aggregate_rolls_per_turn": behavior["rolls_per_turn"],
                        "aggregate_greedy_stop_rate": behavior["greedy_stop_rate"],
                        "selection_score": score,
                        "mean_win_rate": mean_rate,
                    })
                write_csv(output_dir / "validation.csv", evaluation_rows)
                rates_text = ", ".join(
                    f"{name}={report['agents'][0]['win_rate']:.1%}"
                    for name, report in reports.items()
                )
                print(
                    f"Greedy validation at episode {completed}: {rates_text}; "
                    f"selection={score:.1%} ({args.checkpoint_metric}); "
                    f"busts/turn={behavior['busts_per_turn']:.3f}, "
                    f"stops/turn={behavior['stops_per_turn']:.3f}, "
                    f"rolls/turn={behavior['rolls_per_turn']:.3f}, "
                    f"greedy_stop={behavior['greedy_stop_rate']:.1%}"
                )
                if score > best_validation_rate:
                    best_validation_rate = score
                    save_checkpoint(
                        output_dir / "best.pt", online, target, optimizer, config,
                        global_step=global_step, episodes=completed,
                        metadata={
                            **runtime,
                            "validation_selection_score": score,
                            "validation_win_rates": {
                                name: report["agents"][0]["win_rate"]
                                for name, report in reports.items()
                            },
                            "validation_behavior": behavior,
                        },
                    )
                if (completed >= args.collapse_min_episodes
                        and args.collapse_bust_threshold is not None
                        and behavior["busts_per_turn"]
                        > args.collapse_bust_threshold):
                    consecutive_collapsed_validations += 1
                elif completed >= args.collapse_min_episodes:
                    consecutive_collapsed_validations = 0
                if (consecutive_collapsed_validations
                        >= args.collapse_patience):
                    early_stop_reason = (
                        "validation busts/turn remained above "
                        f"{args.collapse_bust_threshold:.3f} for "
                        f"{consecutive_collapsed_validations} evaluations"
                    )
                    save_checkpoint(
                        output_dir / "collapse_early_stop.pt",
                        online, target, optimizer, config,
                        global_step=global_step, episodes=completed,
                        metadata={
                            **runtime,
                            "early_stop_reason": early_stop_reason,
                            "validation_behavior": behavior,
                        },
                    )
                    print(f"Early stopping: {early_stop_reason}")
                while next_eval <= completed:
                    next_eval += args.eval_every
                if early_stop_reason is not None:
                    break

    if actor_backend is not None:
        actor_backend.close()
    if counterfactual_executor is not None:
        counterfactual_executor.shutdown(wait=True)

    target.load_state_dict(online.state_dict())
    save_checkpoint(
        output_dir / "final.pt", online, target, optimizer, config,
        global_step=global_step, episodes=completed,
        metadata={**runtime, "early_stop_reason": early_stop_reason},
    )
    (output_dir / "training_status.json").write_text(json.dumps({
        "requested_episodes": args.episodes,
        "completed_episodes": completed,
        "early_stopped": early_stop_reason is not None,
        "early_stop_reason": early_stop_reason,
        "best_validation_score": best_validation_rate,
    }, indent=2), encoding="utf-8")

    reports = validation_reports(
        online, config, games=args.eval_games,
        seed=args.seed + 20_000_000, max_turns=args.max_turns,
        enhanced_features=args.enhanced_features,
        domain_features=args.domain_features,
        strategic_features=args.strategic_features,
        opponents=args.validation_opponents,
        mc_rollouts=args.validation_mc_rollouts,
        opponent_agents=external_opponents,
        seed_blocks=args.validation_seed_blocks,
        validation_workers=args.validation_workers,
        include_records=True,
    )
    game_records: list[dict[str, object]] = []
    for opponent_name, report in reports.items():
        report["agents"][0]["name"] = f"{args.algorithm}-learned"
        report["agents"][1]["name"] = opponent_name
        for record in report.pop("game_records"):
            game_records.append({"opponent": opponent_name, **record})
    write_evaluation_bundle(
        output_dir / "evaluation_summary.json", reports, args.checkpoint_metric
    )
    write_csv(output_dir / "evaluation_games.csv", game_records)

    # The online model may regress after its best validation checkpoint. Keep
    # the final-model report above for diagnostics, but also evaluate the model
    # that would actually be selected by validation.
    best_path = output_dir / "best.pt"
    best_rate = None
    if best_path.exists():
        checkpoint = load_checkpoint(best_path, torch.device("cpu"))
        best_model = QNetwork(
            config.observation_size, config.action_size, config.hidden_sizes,
            config.network_type,
        )
        best_model.load_state_dict(checkpoint["online_state_dict"])
        best_reports = validation_reports(
            best_model, config, games=args.eval_games,
            seed=args.seed + 20_000_000, max_turns=args.max_turns,
            enhanced_features=args.enhanced_features,
            domain_features=args.domain_features,
            strategic_features=args.strategic_features,
            opponents=args.validation_opponents,
            mc_rollouts=args.validation_mc_rollouts,
            opponent_agents=external_opponents,
            seed_blocks=args.validation_seed_blocks,
            validation_workers=args.validation_workers,
            include_records=True,
        )
        best_game_records: list[dict[str, object]] = []
        for opponent_name, best_report in best_reports.items():
            best_report["agents"][0]["name"] = (
                f"{args.algorithm}-best-validation"
            )
            best_report["agents"][1]["name"] = opponent_name
            for record in best_report.pop("game_records"):
                best_game_records.append({"opponent": opponent_name, **record})
        write_evaluation_bundle(
            output_dir / "best_evaluation_summary.json", best_reports,
            args.checkpoint_metric,
        )
        write_csv(output_dir / "best_evaluation_games.csv", best_game_records)
        best_rate = selection_score(best_reports, args.checkpoint_metric)

    print(f"Saved checkpoint and metrics to {output_dir}")
    print(
        "Final-checkpoint held-out selection score: "
        f"{selection_score(reports, args.checkpoint_metric):.1%}"
    )
    if best_rate is not None:
        print(
            "Best-validation-checkpoint held-out selection score: "
            f"{best_rate:.1%}"
        )


if __name__ == "__main__":
    main()
