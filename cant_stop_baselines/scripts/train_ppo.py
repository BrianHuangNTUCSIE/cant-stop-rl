"""Train a PPO actor-critic for Can't Stop Stop/Continue decisions."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch import nn
from torch.distributions import Categorical

from cant_stop.agents import make_agent
from cant_stop.backend import create_actor_backend
from cant_stop.config import BACKENDS, RuntimeConfig
from cant_stop.evaluation import evaluate
from cant_stop.ppo import (
    PPOActorCritic, PPOAgent, initialize_from_ddqn, initialize_from_ppo,
    joint_action_masks, mask_policy_logits, save_checkpoint,
)
from cant_stop.rl_env import JointAction, StopContinueEnv
from cant_stop.mc_distillation import RolloutConfig, label_state


def parse_mix(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split(","):
        name, weight = item.split(":", 1)
        result[name] = float(weight)
    if not result or any(weight < 0 for weight in result.values()) or sum(result.values()) <= 0:
        raise argparse.ArgumentTypeError("opponent mix must contain non-negative weights")
    total = sum(result.values())
    return {name: weight / total for name, weight in result.items()}


def weighted_name(mix: dict[str, float], rng: random.Random) -> str:
    draw = rng.random()
    cumulative = 0.0
    for name, weight in mix.items():
        cumulative += weight
        if draw < cumulative:
            return name
    return next(reversed(mix))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--backend", choices=BACKENDS, default="python",
        help=("Environment backend: python/python-reference runs environments "
              "in-process; python-multiprocess runs expensive opponent steps "
              "in parallel actor processes"),
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Actor processes for python-multiprocess (default: --num-envs)",
    )
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument("--domain-features", action="store_true", default=True)
    parser.add_argument(
        "--strategic-features", action="store_true",
        help="Use the 80-feature strategic state representation",
    )
    parser.add_argument(
        "--counterfactual-loss-weight", type=float, default=0.0,
        help="Weight for MC counterfactual margin supervision",
    )
    parser.add_argument("--counterfactual-rollouts", type=int, default=8)
    parser.add_argument(
        "--counterfactual-every", type=int, default=8,
        help="Label every Nth Stop/Continue transition (0 disables sampling)",
    )
    parser.add_argument("--counterfactual-future-roll-limit", type=int, default=16)
    parser.add_argument(
        "--pairing-policy", choices=("ga", "learned"), default="ga",
        help="Use fixed GA pairing or jointly learn six masked pairing actions",
    )
    parser.add_argument("--initial-ddqn", type=Path, default=None)
    parser.add_argument(
        "--initial-ppo", type=Path, default=None,
        help="Initialize joint PPO trunk and Continue/Stop outputs from fixed-pairing PPO",
    )
    parser.add_argument(
        "--freeze-decision-policy", action="store_true",
        help=("In learned-pairing mode, train only new pairing inputs and "
              "pairing outputs; preserve transferred Stop/Continue policy"),
    )
    parser.add_argument(
        "--pairing-imitation-examples", type=int, default=20_000,
        help="GA-labeled pair states used to initialize joint pairing outputs",
    )
    parser.add_argument("--pairing-imitation-epochs", type=int, default=3)
    parser.add_argument("--opponent-mix", type=parse_mix,
                        default=parse_mix("mc:0.8,rule28-smart:0.2"))
    parser.add_argument("--training-mc-rollouts", type=int, default=16)
    parser.add_argument("--mc-rollouts", type=int, default=64,
                        help="MC rollouts used for validation and final evaluation")
    parser.add_argument("--mc-future-roll-limit", type=int, default=40)
    parser.add_argument("--validation-opponents", default="mc,rule28-smart")
    parser.add_argument("--validation-games", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="Print training progress every N completed episodes")
    parser.add_argument("--seed", type=int, default=450)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def evaluate_policy(policy: PPOAgent, opponent_names: list[str], games: int, seed: int,
                    mc_rollouts: int, mc_future_roll_limit: int) -> dict[str, object]:
    reports: dict[str, object] = {}
    for offset, name in enumerate(opponent_names):
        opponent = make_agent(name, mc_rollouts, mc_future_roll_limit)
        reports[name] = evaluate(
            policy, opponent, games=games, seed=seed + offset * 100_003,
            max_total_turns=2000,
        )
    return reports


def validation_behavior(reports: dict[str, dict]) -> dict[str, float]:
    """Aggregate greedy PPO behavior over all validation opponents."""
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


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return value-function explained variance, or zero for constant targets."""
    target_variance = targets.var(unbiased=False)
    if float(target_variance) <= 1e-8:
        return 0.0
    residual_variance = (targets - predictions).var(unbiased=False)
    return float(1.0 - residual_variance / target_variance)


def rank_validation_checkpoints(entries: list[dict], limit: int = 3) -> list[dict]:
    """Rank robust checkpoints by minimum validation rate, then earlier episode."""
    return sorted(
        entries,
        key=lambda entry: (-float(entry["selection_score"]), int(entry["episodes"])),
    )[:limit]


def collect_pairing_imitation_examples(
    count: int, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect deterministic GA labels at learner pair phases."""
    if count < 1:
        raise ValueError("imitation example count must be positive")
    teacher = make_agent("rule28-smart")
    opponent = make_agent("rule28-smart")
    teacher_rng = random.Random(seed ^ 0xC001D00D)
    observations: list[list[float]] = []
    labels: list[int] = []
    masks: list[list[bool]] = []
    episode = 0
    env = StopContinueEnv(
        domain_features=True, learn_pairing=True, opponent=opponent, seed=seed,
    )
    observation, _ = env.reset()
    while len(observations) < count:
        if observation is None:
            episode += 1
            env = StopContinueEnv(
                domain_features=True, learn_pairing=True, opponent=opponent,
                learner_player=episode % 2, seed=seed + episode,
            )
            observation, _ = env.reset()
            continue
        if env.game.phase == "pair":
            ordered = sorted(env.game.available)
            selected = teacher.choose_move(
                env.game.snapshot(), ordered, teacher_rng
            )
            label = ordered.index(selected)
            observations.append(observation)
            labels.append(label)
            masks.append(env.legal_action_mask())
            action = label
        else:
            stop = teacher.should_stop(env.game.snapshot(), teacher_rng)
            action = int(JointAction.STOP if stop else JointAction.CONTINUE)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation = None
    return (
        torch.tensor(observations, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
    )


def initialize_pairing_by_imitation(
    model: PPOActorCritic, *, examples: int, epochs: int, seed: int,
    minibatch_size: int = 256,
) -> dict[str, float]:
    """Fit new pairing inputs/outputs while preserving PPO-451 decisions."""
    if model.observation_size != 86 or model.action_size != 8:
        raise ValueError("pairing imitation requires a joint PPO model")
    if epochs < 1 or minibatch_size < 1:
        raise ValueError("imitation epochs and minibatch size must be positive")
    observations, labels, masks = collect_pairing_imitation_examples(examples, seed)
    first_layer = next(
        layer for layer in model.trunk if isinstance(layer, nn.Linear)
    )
    optimizer = torch.optim.Adam(
        (first_layer.weight, model.policy_head.weight, model.policy_head.bias),
        lr=1e-3,
    )
    generator = torch.Generator().manual_seed(seed ^ 0x51A7E)
    for _ in range(epochs):
        order = torch.randperm(examples, generator=generator)
        for start in range(0, examples, minibatch_size):
            indices = order[start:start + minibatch_size]
            hidden = model.trunk(observations[indices])
            logits = mask_policy_logits(model.policy_head(hidden), masks[indices])
            loss = nn.functional.cross_entropy(logits, labels[indices])
            model.zero_grad(set_to_none=True)
            loss.backward()
            # Only the 18 new input columns and six pairing output rows move.
            # Decision observations set those inputs to zero, so this preserves
            # the transferred PPO-451 Continue/Stop policy exactly.
            assert first_layer.weight.grad is not None
            assert model.policy_head.weight.grad is not None
            assert model.policy_head.bias.grad is not None
            first_layer.weight.grad[:, :68].zero_()
            model.policy_head.weight.grad[6:8].zero_()
            model.policy_head.bias.grad[6:8].zero_()
            optimizer.step()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        logits = mask_policy_logits(model(observations)[0], masks)
        accuracy = float((logits.argmax(dim=1) == labels).float().mean())
        loss = float(nn.functional.cross_entropy(logits, labels))
    return {"examples": float(examples), "epochs": float(epochs),
            "accuracy": accuracy, "loss": loss}


def pairing_only_parameters(model: PPOActorCritic) -> list[nn.Parameter]:
    """Return the parameters that can change without changing decisions.

    Pairing observations have nonzero values only in the final 18 input
    columns, and decision observations have zeros there. Restricting PPO to
    these columns plus pairing logits therefore keeps the transferred
    Stop/Continue policy exactly fixed.
    """
    if model.observation_size != 86 or model.action_size != 8:
        raise ValueError("pairing-only optimization requires an 86/8 PPO model")
    first_layer = next(
        layer for layer in model.trunk if isinstance(layer, nn.Linear)
    )
    return [
        first_layer.weight,
        model.policy_head.weight,
        model.policy_head.bias,
    ]


def restrict_pairing_gradients(model: PPOActorCritic) -> None:
    """Mask PPO gradients so frozen decision behavior cannot change."""
    first_layer = next(
        layer for layer in model.trunk if isinstance(layer, nn.Linear)
    )
    if first_layer.weight.grad is not None:
        first_layer.weight.grad[:, :68].zero_()
    if model.policy_head.weight.grad is not None:
        model.policy_head.weight.grad[6:8].zero_()
    if model.policy_head.bias.grad is not None:
        model.policy_head.bias.grad[6:8].zero_()


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.num_envs < 1 or args.rollout_steps < 1:
        raise SystemExit("episodes, num-envs, and rollout-steps must be positive")
    if args.minibatch_size < 1 or args.ppo_epochs < 1:
        raise SystemExit("minibatch-size and ppo-epochs must be positive")
    if args.counterfactual_loss_weight < 0 or args.counterfactual_rollouts < 1:
        raise SystemExit("counterfactual weight must be non-negative and rollouts positive")
    if args.counterfactual_every < 0:
        raise SystemExit("counterfactual-every must be non-negative")
    if args.num_workers is not None and args.num_workers < 1:
        raise SystemExit("num-workers must be positive")
    if args.eval_every < 1 or args.progress_every < 1:
        raise SystemExit("eval-every and progress-every must be positive")
    if args.initial_ddqn is not None and args.initial_ppo is not None:
        raise SystemExit("choose at most one of --initial-ddqn and --initial-ppo")
    if args.initial_ppo is not None and args.pairing_policy == "learned":
        raise SystemExit("--initial-ppo transfer is supported for fixed GA pairing only")
    if args.freeze_decision_policy and (
        args.pairing_policy != "learned" or args.initial_ppo is None
    ):
        raise SystemExit(
            "--freeze-decision-policy requires learned pairing and --initial-ppo"
        )
    if args.initial_ddqn is not None and args.pairing_policy == "learned":
        raise SystemExit("--initial-ddqn is only supported with fixed GA pairing")
    if args.pairing_imitation_examples < 0 or args.pairing_imitation_epochs < 1:
        raise SystemExit("imitation examples must be non-negative and epochs positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    mix_rng = random.Random(args.seed ^ 0xA5A5A5A5)
    hidden_sizes = tuple(int(value) for value in args.hidden_sizes.split(","))
    validation_opponents = [value.strip() for value in args.validation_opponents.split(",") if value.strip()]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

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

    learn_pairing = args.pairing_policy == "learned"
    observation_base = 80 if args.strategic_features else 68
    observation_size = observation_base + (18 if learn_pairing else 0)
    action_size = 8 if learn_pairing else 2
    model = PPOActorCritic(observation_size, hidden_sizes, action_size)
    model.strategic_features = args.strategic_features
    if args.initial_ddqn is not None:
        initialize_from_ddqn(model, args.initial_ddqn)
    initial_ppo_config = None
    if args.initial_ppo is not None:
        initial_ppo_config = initialize_from_ppo(model, args.initial_ppo)
    imitation_metrics = None
    if learn_pairing and args.pairing_imitation_examples:
        imitation_metrics = initialize_pairing_by_imitation(
            model, examples=args.pairing_imitation_examples,
            epochs=args.pairing_imitation_epochs, seed=args.seed,
            minibatch_size=args.minibatch_size,
        )
        print(f"Pairing imitation: {imitation_metrics}", flush=True)
    trainable_parameters = (
        pairing_only_parameters(model)
        if args.freeze_decision_policy else list(model.parameters())
    )
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
    device = torch.device("cpu")
    model.to(device)

    config = {
        "algorithm": "ppo",
        "observation_size": observation_size,
        "action_size": action_size,
        "hidden_sizes": hidden_sizes,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_epsilon": args.clip_epsilon,
        "entropy_coefficient": args.entropy_coefficient,
        "value_coefficient": args.value_coefficient,
        "max_grad_norm": args.max_grad_norm,
        "rollout_steps": args.rollout_steps,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "domain_features": not args.strategic_features,
        "strategic_features": args.strategic_features,
        "counterfactual_loss_weight": args.counterfactual_loss_weight,
        "counterfactual_rollouts": args.counterfactual_rollouts,
        "counterfactual_every": args.counterfactual_every,
        "counterfactual_future_roll_limit": args.counterfactual_future_roll_limit,
        "pairing_policy": args.pairing_policy,
        "freeze_decision_policy": args.freeze_decision_policy,
        "opponent_mix": args.opponent_mix,
        "training_mc_rollouts": args.training_mc_rollouts,
        "evaluation_mc_rollouts": args.mc_rollouts,
        "mc_future_roll_limit": args.mc_future_roll_limit,
        "initial_ddqn": str(args.initial_ddqn) if args.initial_ddqn else None,
        "initial_ppo": str(args.initial_ppo) if args.initial_ppo else None,
        "initial_ppo_config": initial_ppo_config,
        "pairing_imitation_examples": args.pairing_imitation_examples,
        "pairing_imitation_epochs": args.pairing_imitation_epochs,
        "pairing_imitation_metrics": imitation_metrics,
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        **runtime_config.metadata(),
        "num_envs": args.num_envs,
        "active_environment_count": active_environment_count,
        "episodes": args.episodes,
        "validation_opponents": validation_opponents,
        "validation_games": args.validation_games,
        "final_evaluation_games": args.eval_games,
        "evaluation_interval_episodes": args.eval_every,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, default=list), encoding="utf-8")

    # Built-in training agents are stateless; all randomness is supplied by
    # each environment's independent policy RNG. Reusing them avoids rebuilding
    # MC and pairing policy objects for every episode.
    pairing_policy = make_agent(
        "rule28-smart", args.training_mc_rollouts, args.mc_future_roll_limit
    )
    training_opponents = {
        name: make_agent(name, args.training_mc_rollouts, args.mc_future_roll_limit)
        for name in args.opponent_mix
    }

    def make_training_env(episode_id: int) -> StopContinueEnv:
        opponent_name = weighted_name(args.opponent_mix, mix_rng)
        return StopContinueEnv(
            domain_features=not args.strategic_features,
            strategic_features=args.strategic_features,
            learner_player=episode_id % 2,
            pairing_policy=pairing_policy,
            opponent=training_opponents[opponent_name],
            seed=args.seed + episode_id,
            learn_pairing=learn_pairing,
        )

    episode_ids = list(range(active_environment_count))
    next_episode = active_environment_count
    completed = 0
    envs: list[StopContinueEnv] = []
    observations: list[list[float] | None] = []
    actor_backend = create_actor_backend(runtime_config)
    initial_envs = [make_training_env(episode_id) for episode_id in episode_ids]
    if actor_backend is None:
        for episode_id, env in zip(episode_ids, initial_envs):
            observation, _ = env.reset(seed=args.seed + episode_id)
            envs.append(env)
            observations.append(observation)
    else:
        initial_states = actor_backend.load_basic_many(list(enumerate(initial_envs)))
        for index in range(active_environment_count):
            observation = initial_states[index]
            observations.append(observation)

    def fill_observationless_slots() -> None:
        """Replace games that ended before the learner's first action.

        A sufficiently long first turn can let seat 0 win before a seat-1
        learner receives any observation. Such a game is a completed episode,
        but contributes no PPO transition.
        """
        nonlocal completed, next_episode
        for index in range(active_environment_count):
            while observations[index] is None and completed < args.episodes:
                completed += 1
                if completed >= args.episodes:
                    observations[index] = [0.0] * observation_size
                    break
                episode_id = next_episode
                next_episode += 1
                replacement = make_training_env(episode_id)
                if actor_backend is None:
                    envs[index] = replacement
                    observation, _ = replacement.reset(
                        seed=args.seed + episode_id
                    )
                else:
                    observation = actor_backend.load_basic_many(
                        [(index, replacement)]
                    )[index]
                observations[index] = observation

    fill_observationless_slots()
    if completed >= args.episodes:
        raise RuntimeError(
            "all requested episodes ended before the learner received an action"
        )
    assert all(observation is not None for observation in observations)
    print(
        f"Environment backend: {runtime_config.canonical_backend}; "
        f"active environments={active_environment_count}; "
        f"workers={runtime_config.worker_count}",
        flush=True,
    )

    best_score = -1.0
    update_count = 0
    started = time.monotonic()
    counterfactual_labels = 0
    next_progress = args.progress_every
    next_evaluation = args.eval_every
    validation_path = output / "validation.csv"
    metrics_path = output / "training_metrics.csv"
    validation_checkpoints: list[dict] = []
    with (
        validation_path.open("w", newline="", encoding="utf-8") as validation_file,
        metrics_path.open("w", newline="", encoding="utf-8") as metrics_file,
    ):
        validation_writer = csv.writer(validation_file)
        validation_writer.writerow((
            "episodes", "opponent", "games", "wins", "win_rate",
            "selection_score", "busts_per_turn", "stops_per_turn",
            "rolls_per_turn", "greedy_stop_rate",
            "aggregate_busts_per_turn", "aggregate_stops_per_turn",
            "aggregate_rolls_per_turn", "aggregate_greedy_stop_rate",
        ))
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow((
            "episodes", "update", "transitions", "policy_loss", "value_loss",
            "entropy", "approx_kl", "clip_fraction", "gradient_norm",
            "explained_variance", "mean_return", "return_std", "learning_rate",
        ))

        while completed < args.episodes:
            state_chunks: list[torch.Tensor] = []
            action_chunks: list[torch.Tensor] = []
            mask_chunks: list[torch.Tensor] = []
            old_log_chunks: list[torch.Tensor] = []
            value_chunks: list[torch.Tensor] = []
            counterfactual_chunks: list[torch.Tensor] = []
            counterfactual_mask_chunks: list[torch.Tensor] = []
            rewards: list[float] = []
            dones: list[bool] = []
            counterfactual_seen = 0
            for _ in range(args.rollout_steps):
                assert all(observation is not None for observation in observations)
                state_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits, state_values = model(state_tensor)
                    legal_masks = (
                        joint_action_masks(state_tensor) if learn_pairing
                        else torch.ones_like(logits, dtype=torch.bool)
                    )
                    logits = mask_policy_logits(logits, legal_masks)
                    distribution = Categorical(logits=logits)
                    batch_actions = distribution.sample()
                    batch_log_probs = distribution.log_prob(batch_actions)
                selected_actions = batch_actions.tolist()
                # Labels are computed from the immutable pre-action state.  We
                # deliberately keep this synchronous and deterministic for the
                # first Phase-A implementation; the rollout cadence limits the
                # cost and makes the label stream easy to audit.
                cf_targets_step: list[float] = []
                cf_masks_step: list[float] = []
                for index in range(active_environment_count):
                    use_label = (
                        args.counterfactual_loss_weight > 0.0
                        and args.counterfactual_every > 0
                        and not learn_pairing
                        and actor_backend is None
                        and envs[index].game.phase == "decision"
                        and counterfactual_seen % args.counterfactual_every == 0
                    )
                    if use_label:
                        config_cf = RolloutConfig(
                            rollouts=args.counterfactual_rollouts,
                            opponent_rollouts=min(8, args.training_mc_rollouts),
                            future_roll_limit=args.counterfactual_future_roll_limit,
                        )
                        q_continue, q_stop = label_state(
                            envs[index].game.snapshot(), config=config_cf,
                            seed=args.seed + update_count * 100003 + counterfactual_seen,
                        )
                        counterfactual_labels += 1
                        cf_targets_step.append(float(q_continue - q_stop))
                        cf_masks_step.append(1.0)
                    else:
                        cf_targets_step.append(0.0)
                        cf_masks_step.append(0.0)
                    if (not learn_pairing and actor_backend is None
                            and envs[index].game.phase == "decision"):
                        counterfactual_seen += 1
                processed = 0
                if actor_backend is None:
                    step_results = [env.step(selected_actions[index]) for index, env in enumerate(envs)]
                else:
                    remote_results = actor_backend.step_basic_many([
                        (index, selected_actions[index])
                        for index in range(active_environment_count)
                    ])
                    step_results = [
                        (
                            *remote_results[index],
                            None,
                        )
                        for index in range(active_environment_count)
                    ]
                reloads: list[tuple[int, StopContinueEnv]] = []
                for index, result in enumerate(step_results):
                    next_observation, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                    rewards.append(float(reward))
                    dones.append(done)
                    processed += 1
                    if done:
                        completed += 1
                        if completed < args.episodes:
                            episode_id = next_episode
                            next_episode += 1
                            replacement = make_training_env(episode_id)
                            if actor_backend is None:
                                envs[index] = replacement
                                observations[index], _ = replacement.reset(
                                    seed=args.seed + episode_id
                                )
                            else:
                                reloads.append((index, replacement))
                    else:
                        assert next_observation is not None
                        observations[index] = next_observation
                    if completed >= args.episodes:
                        break
                if actor_backend is not None and reloads:
                    reloaded = actor_backend.load_basic_many(reloads)
                    for index in reloaded:
                        observation = reloaded[index]
                        observations[index] = observation
                fill_observationless_slots()
                # Keep rollout data as tensors from the outset. The old path
                # expanded every observation and output into Python scalars,
                # then copied all of them back into tensors before optimization.
                state_chunks.append(state_tensor[:processed])
                action_chunks.append(batch_actions[:processed])
                mask_chunks.append(legal_masks[:processed])
                old_log_chunks.append(batch_log_probs[:processed])
                value_chunks.append(state_values[:processed])
                counterfactual_chunks.append(torch.tensor(
                    cf_targets_step[:processed], dtype=torch.float32, device=device
                ))
                counterfactual_mask_chunks.append(torch.tensor(
                    cf_masks_step[:processed], dtype=torch.float32, device=device
                ))
                if completed >= args.episodes:
                    break

            count = len(rewards)
            if count == 0:
                break
            with torch.no_grad():
                final_values = model(torch.tensor(observations, dtype=torch.float32, device=device))[1].tolist()
            values = torch.cat(value_chunks).tolist()
            advantage = [0.0] * count
            returns = [0.0] * count
            # Transitions are stored environment-major within each time step.
            for time_index in range((count + active_environment_count - 1) // active_environment_count - 1, -1, -1):
                for env_index in range(active_environment_count - 1, -1, -1):
                    flat = time_index * active_environment_count + env_index
                    if flat >= count:
                        continue
                    next_flat = flat + active_environment_count
                    next_value = final_values[env_index] if next_flat >= count else values[next_flat]
                    next_nonterminal = 0.0 if dones[flat] else 1.0
                    delta = rewards[flat] + args.gamma * next_value * next_nonterminal - values[flat]
                    following = 0.0 if dones[flat] else (advantage[next_flat] if next_flat < count else 0.0)
                    advantage[flat] = delta + args.gamma * args.gae_lambda * following
                    returns[flat] = advantage[flat] + values[flat]
            state_batch = torch.cat(state_chunks)
            action_batch = torch.cat(action_chunks)
            mask_batch = torch.cat(mask_chunks)
            old_log_batch = torch.cat(old_log_chunks)
            return_batch = torch.tensor(returns, dtype=torch.float32, device=device)
            advantage_batch = torch.tensor(advantage, dtype=torch.float32, device=device)
            counterfactual_batch = torch.cat(counterfactual_chunks)
            counterfactual_mask_batch = torch.cat(counterfactual_mask_chunks)
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)
            indices = torch.arange(count, device=device)
            weighted_policy_loss = 0.0
            weighted_value_loss = 0.0
            weighted_entropy = 0.0
            weighted_approx_kl = 0.0
            weighted_clip_fraction = 0.0
            diagnostic_samples = 0
            gradient_norm_sum = 0.0
            minibatch_count = 0
            for _ in range(args.ppo_epochs):
                shuffled = indices[torch.randperm(count, device=device)]
                for start in range(0, count, args.minibatch_size):
                    batch_indices = shuffled[start:start + args.minibatch_size]
                    logits, predicted_values = model(state_batch[batch_indices])
                    logits = mask_policy_logits(logits, mask_batch[batch_indices])
                    distribution = Categorical(logits=logits)
                    log_probs = distribution.log_prob(action_batch[batch_indices])
                    ratio = (log_probs - old_log_batch[batch_indices]).exp()
                    advantages = advantage_batch[batch_indices]
                    clipped = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
                    policy_loss = -torch.minimum(ratio * advantages, clipped).mean()
                    value_loss = 0.5 * (predicted_values - return_batch[batch_indices]).pow(2).mean()
                    counterfactual_loss = torch.zeros((), device=device)
                    if args.counterfactual_loss_weight > 0.0:
                        cf_mask = counterfactual_mask_batch[batch_indices]
                        if bool(cf_mask.any()):
                            # For the fixed two-action policy, logit_continue -
                            # logit_stop is a smooth local preference whose
                            # sign matches the action distribution.
                            predicted_margin = logits[:, 0] - logits[:, 1]
                            counterfactual_loss = (
                                (predicted_margin - counterfactual_batch[batch_indices])
                                .pow(2) * cf_mask
                            ).sum() / cf_mask.sum()
                    entropy = distribution.entropy().mean()
                    loss = (policy_loss + args.value_coefficient * value_loss
                            - args.entropy_coefficient * entropy
                            + args.counterfactual_loss_weight * counterfactual_loss)
                    model.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.freeze_decision_policy:
                        restrict_pairing_gradients(model)
                    gradient_norm = nn.utils.clip_grad_norm_(
                        trainable_parameters, args.max_grad_norm
                    )
                    optimizer.step()
                    with torch.no_grad():
                        log_ratio = log_probs - old_log_batch[batch_indices]
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_fraction = (
                            (ratio - 1.0).abs() > args.clip_epsilon
                        ).float().mean()
                    batch_count = int(batch_indices.numel())
                    weighted_policy_loss += policy_loss.detach().item() * batch_count
                    weighted_value_loss += value_loss.detach().item() * batch_count
                    weighted_entropy += entropy.detach().item() * batch_count
                    weighted_approx_kl += approx_kl.item() * batch_count
                    weighted_clip_fraction += clip_fraction.item() * batch_count
                    diagnostic_samples += batch_count
                    gradient_norm_sum += gradient_norm.item()
                    minibatch_count += 1
            update_count += 1
            with torch.no_grad():
                _, fitted_values = model(state_batch)
                value_explained_variance = explained_variance(
                    fitted_values, return_batch
                )
            metrics_writer.writerow((
                completed,
                update_count,
                count,
                weighted_policy_loss / diagnostic_samples,
                weighted_value_loss / diagnostic_samples,
                weighted_entropy / diagnostic_samples,
                weighted_approx_kl / diagnostic_samples,
                weighted_clip_fraction / diagnostic_samples,
                gradient_norm_sum / minibatch_count,
                value_explained_variance,
                float(return_batch.mean()),
                float(return_batch.std(unbiased=False)),
                optimizer.param_groups[0]["lr"],
            ))
            metrics_file.flush()

            now = time.monotonic()
            if completed >= next_progress or completed >= args.episodes:
                elapsed = now - started
                episode_rate = completed / elapsed if elapsed > 0 else 0.0
                update_rate = update_count / elapsed if elapsed > 0 else 0.0
                remaining = max(0, args.episodes - completed)
                eta = remaining / episode_rate if episode_rate > 0 else 0.0
                print(
                    f"progress: {min(completed, args.episodes)}/{args.episodes} episodes "
                    f"| update {update_count} | {episode_rate:.1f} ep/s "
                    f"| {update_rate:.2f} updates/s | elapsed {elapsed / 60:.1f} min "
                    f"| ETA {eta / 60:.1f} min | cf_labels {counterfactual_labels}",
                    flush=True,
                )
                while next_progress <= completed:
                    next_progress += args.progress_every

            # Rollouts can finish many episodes at once, so use threshold
            # crossing rather than ``completed % eval_every == 0``.
            if completed >= next_evaluation:
                policy = PPOAgent(model)
                reports = evaluate_policy(policy, validation_opponents, args.validation_games,
                                          1_000_000 + args.seed + completed,
                                          args.mc_rollouts, args.mc_future_roll_limit)
                rates = {name: report["agents"][0]["win_rate"] for name, report in reports.items()}
                score = min(rates.values())
                aggregate_behavior = validation_behavior(reports)
                for name in validation_opponents:
                    report = reports[name]
                    agent = report["agents"][0]
                    turns = agent["total_turns"]
                    rolls = agent["total_rolls"]
                    busts = agent["total_busts"]
                    stops = agent["total_stops"]
                    decisions = rolls - busts
                    validation_writer.writerow((
                        completed, name, args.validation_games, agent["wins"],
                        rates[name], score,
                        busts / turns if turns else 0.0,
                        stops / turns if turns else 0.0,
                        rolls / turns if turns else 0.0,
                        stops / decisions if decisions else 0.0,
                        aggregate_behavior["busts_per_turn"],
                        aggregate_behavior["stops_per_turn"],
                        aggregate_behavior["rolls_per_turn"],
                        aggregate_behavior["greedy_stop_rate"],
                    ))
                validation_file.flush()
                checkpoint_path = (
                    output / "checkpoints" / f"episode_{completed}.pt"
                )
                save_checkpoint(
                    checkpoint_path, model, optimizer, config,
                    episodes=completed, updates=update_count,
                )
                validation_checkpoints.append({
                    "episodes": completed,
                    "updates": update_count,
                    "selection_score": score,
                    "win_rates": rates,
                    "behavior": aggregate_behavior,
                    "checkpoint": str(checkpoint_path.relative_to(output)),
                })
                top_checkpoints = rank_validation_checkpoints(
                    validation_checkpoints
                )
                for rank, entry in enumerate(top_checkpoints, start=1):
                    shutil.copy2(
                        output / entry["checkpoint"],
                        output / f"top_{rank}.pt",
                    )
                (output / "top_checkpoints.json").write_text(
                    json.dumps({
                        "selection_metric": "minimum_win_rate",
                        "checkpoints": top_checkpoints,
                    }, indent=2),
                    encoding="utf-8",
                )
                if score > best_score:
                    best_score = score
                    shutil.copy2(checkpoint_path, output / "best.pt")
                print(
                    f"episodes={completed} validation={rates} best={best_score:.3f} "
                    f"busts/turn={aggregate_behavior['busts_per_turn']:.3f} "
                    f"greedy_stop={aggregate_behavior['greedy_stop_rate']:.1%}",
                    flush=True,
                )
                while next_evaluation <= completed:
                    next_evaluation += args.eval_every

    if actor_backend is not None:
        actor_backend.close()
    save_checkpoint(output / "final.pt", model, optimizer, config,
                    episodes=completed, updates=update_count)
    policy = PPOAgent(model)
    reports = evaluate_policy(policy, validation_opponents, args.eval_games,
                              2_000_000 + args.seed, args.mc_rollouts, args.mc_future_roll_limit)
    (output / "evaluation_summary.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    print(f"completed={completed} updates={update_count} elapsed={time.monotonic() - started:.1f}s")
    for name, report in reports.items():
        print(f"{name}: {report['agents'][0]['win_rate']:.3%}")


if __name__ == "__main__":
    main()
