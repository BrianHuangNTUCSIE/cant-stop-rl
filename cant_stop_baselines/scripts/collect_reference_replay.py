"""Collect frozen-policy transitions for replay-preserving fine-tuning."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cant_stop_core.agents import make_agent
from cant_stop_core.classical_ml import ClassicalMLAgent
from cant_stop_core.dqn import NStepAccumulator, QNetwork, choose_actions, load_checkpoint
from cant_stop_core.rl_env import StopContinueEnv
from scripts.train_dqn import parse_opponent_mix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=int, default=375_000)
    parser.add_argument("--seed", type=int, default=447)
    parser.add_argument("--max-turns", type=int, default=2_000)
    parser.add_argument(
        "--opponent-mix", type=parse_opponent_mix,
        default=parse_opponent_mix("rule28-smart:0.4,mc:0.4,ridge-domain:0.2"),
    )
    parser.add_argument("--ridge-opponent", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.transitions < 1:
        raise SystemExit("transitions must be positive")

    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = dict(checkpoint["config"])
    model = QNetwork(
        int(config["observation_size"]), int(config["action_size"]),
        tuple(config["hidden_sizes"]), config["network_type"],
    ).to(device)
    model.load_state_dict(checkpoint["online_state_dict"])
    model.eval()
    ridge = ClassicalMLAgent.from_file(args.ridge_opponent)
    rng = random.Random(args.seed ^ 0xBADC0DE)
    states: list[list[float]] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[list[float] | None] = []
    dones: list[bool] = []
    n_steps: list[int] = []
    counterfactuals: list[list[float]] = []
    counterfactual_masks: list[bool] = []
    episode = 0
    while len(states) < args.transitions:
        draw = rng.random()
        cumulative = 0.0
        selected = args.opponent_mix[-1][0]
        for name, weight in args.opponent_mix:
            cumulative += weight
            if draw < cumulative:
                selected = name
                break
        opponent = ridge if selected == "ridge-domain" else make_agent(selected)
        env = StopContinueEnv(
            enhanced_features=bool(config.get("observation_size") == 60),
            domain_features=bool(config.get("observation_size") >= 68),
            learner_player=episode % 2,
            pairing_policy=make_agent("rule28-smart"),
            opponent=opponent,
            seed=args.seed + episode,
            max_total_turns=args.max_turns,
            learn_pairing=False,
        )
        observation, _ = env.reset()
        accumulator = NStepAccumulator(int(config.get("n_step", 5)), float(config["gamma"]))
        while observation is not None and len(states) < args.transitions:
            legal = env.legal_action_mask() if env.game.phase == "pair" else None
            action = choose_actions(
                model, [observation], 0.0, rng, device,
                [legal] if legal is not None else None,
            )[0]
            next_observation, reward, terminated, truncated, _ = env.step(action)
            for transition in accumulator.append(
                list(observation), action, float(reward),
                None if next_observation is None else list(next_observation),
                bool(terminated or truncated),
            ):
                states.append(list(transition.state))
                actions.append(transition.action)
                rewards.append(float(transition.reward))
                next_states.append(transition.next_state)
                dones.append(transition.done)
                n_steps.append(transition.n_steps)
                counterfactuals.append([0.0, 0.0])
                counterfactual_masks.append(False)
                if len(states) >= args.transitions:
                    break
            observation = next_observation
            if terminated or truncated:
                break
        episode += 1
        if episode % 25 == 0:
            print(f"episodes={episode} transitions={len(states):,}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states": states, "actions": actions, "rewards": rewards,
        "next_states": next_states, "dones": dones, "n_steps": n_steps,
        "counterfactuals": counterfactuals,
        "counterfactual_masks": counterfactual_masks,
        "seed": args.seed, "episodes": episode,
    }, args.output)
    print(f"Saved {len(states):,} transitions to {args.output}")


if __name__ == "__main__":
    main()
