"""Generate rollout-labelled Stop/Continue states for supervised ML."""
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
import platform
import random
import time
from pathlib import Path

from cant_stop.agents import Rule28Agent
from cant_stop.game import Game
from cant_stop.model import State
from cant_stop.rl_env import StopContinueAction


FEATURE_COLUMNS = [f"x{index:02d}" for index in range(68)]


def game_from_decision_state(state: State) -> Game:
    """Reconstruct the outcome-relevant game fields from a decision snapshot."""
    game = Game()
    game.permanent = [list(state.permanent[0]), list(state.permanent[1])]
    game.owner = list(state.owner)
    game.active = state.active_dict()
    game.current_player = state.current_player
    game.rolls_this_turn = state.rolls_this_turn
    game.phase = "decision"
    return game


def finish_game(game: Game, seed: int, max_future_turns: int) -> int | None:
    """Play from an arbitrary phase with a fixed smart Rule-of-28 policy."""
    policy = Rule28Agent(pair_heuristic="ga", smart=True)
    dice_rng = random.Random(seed)
    policy_rngs = (
        random.Random(seed ^ 0x1234ABCD),
        random.Random(seed ^ 0xFEDC5678),
    )
    turns_started = 0
    while game.winner is None:
        if game.phase == "roll" and game.rolls_this_turn == 0:
            if turns_started >= max_future_turns:
                return None
            turns_started += 1
        player = game.current_player
        if game.phase == "roll":
            moves = game.roll(dice_rng)
            if not moves:
                continue
        if game.phase == "pair":
            state = game.snapshot()
            game.apply_move(policy.choose_move(state, game.available, policy_rngs[player]))
        if game.phase == "decision":
            state = game.snapshot()
            if policy.should_stop(state, policy_rngs[player]):
                game.stop()
            else:
                game.continue_turn()
    return game.winner


def branch_outcome(state: State, action: StopContinueAction, seed: int,
                   max_future_turns: int) -> float:
    game = game_from_decision_state(state)
    target_player = state.current_player
    if action is StopContinueAction.STOP:
        game.stop()
    else:
        game.continue_turn()
    winner = finish_game(game, seed, max_future_turns)
    if winner is None:
        return 0.5
    return 1.0 if winner == target_player else 0.0


def label_state(state: State, *, rollouts: int, seed: int,
                max_future_turns: int) -> tuple[float, float]:
    """Estimate both action values with paired common-random-number rollouts."""
    stop_total = 0.0
    continue_total = 0.0
    for index in range(rollouts):
        rollout_seed = seed + index
        stop_total += branch_outcome(
            state, StopContinueAction.STOP, rollout_seed, max_future_turns
        )
        continue_total += branch_outcome(
            state, StopContinueAction.CONTINUE, rollout_seed, max_future_turns
        )
    return stop_total / rollouts, continue_total / rollouts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--max-states", type=int, default=10_000)
    parser.add_argument("--sample-probability", type=float, default=0.5)
    parser.add_argument("--exploration-probability", type=float, default=0.5,
                        help="Chance of a random Stop/Continue collection action")
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=541)
    parser.add_argument("--max-turns", type=int, default=2_000)
    parser.add_argument("--max-future-turns", type=int, default=500)
    parser.add_argument("--output", type=Path,
                        default=Path("data/generated/classical_ml_dataset.csv"))
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.games, args.max_states, args.rollouts, args.max_turns,
           args.max_future_turns, args.log_every) < 1:
        raise SystemExit("game, state, rollout, turn, and logging counts must be positive")
    if not 0.0 < args.sample_probability <= 1.0:
        raise SystemExit("sample-probability must be in (0, 1]")
    if not 0.0 <= args.exploration_probability <= 1.0:
        raise SystemExit("exploration-probability must be in [0, 1]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    fields = [
        "game_id", "game_seed", "decision_index", "player", "rolls_this_turn",
        "rollouts", "q_stop", "q_continue", "continue_margin", "label",
        *FEATURE_COLUMNS,
    ]
    smart = Rule28Agent(pair_heuristic="ga", smart=True)
    collected = 0
    completed_games = 0
    started = time.perf_counter()

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for game_id in range(args.games):
            if collected >= args.max_states:
                break
            game_seed = args.seed + game_id
            dice_rng = random.Random(game_seed)
            behavior_rngs = (
                random.Random(game_seed ^ 0xA11CE001),
                random.Random(game_seed ^ 0xA11CE002),
            )
            sampling_rng = random.Random(game_seed ^ 0x51A7E000)
            game = Game()
            decisions = 0
            while game.winner is None and sum(game.turns) < args.max_turns:
                player = game.current_player
                if game.phase == "roll":
                    moves = game.roll(dice_rng)
                    if not moves:
                        continue
                if game.phase == "pair":
                    state = game.snapshot()
                    move = smart.choose_move(state, game.available, behavior_rngs[player])
                    game.apply_move(move)
                if game.phase == "decision":
                    state = game.snapshot()
                    if (collected < args.max_states
                            and sampling_rng.random() < args.sample_probability):
                        label_seed = (
                            args.seed + 1_000_000_000
                            + game_id * 1_000_003 + decisions * 10_007
                        )
                        q_stop, q_continue = label_state(
                            state, rollouts=args.rollouts, seed=label_seed,
                            max_future_turns=args.max_future_turns,
                        )
                        margin = q_continue - q_stop
                        # Conservative tie break: bank when estimated values tie.
                        label = int(
                            StopContinueAction.CONTINUE if margin > 0.0
                            else StopContinueAction.STOP
                        )
                        features = state.features(domain=True)
                        row: dict[str, object] = {
                            "game_id": game_id,
                            "game_seed": game_seed,
                            "decision_index": decisions,
                            "player": player,
                            "rolls_this_turn": state.rolls_this_turn,
                            "rollouts": args.rollouts,
                            "q_stop": q_stop,
                            "q_continue": q_continue,
                            "continue_margin": margin,
                            "label": label,
                        }
                        row.update(zip(FEATURE_COLUMNS, features))
                        writer.writerow(row)
                        collected += 1

                    rng = behavior_rngs[player]
                    random_action = rng.random() < args.exploration_probability
                    should_stop = (
                        rng.random() < 0.5 if random_action
                        else smart.should_stop(state, rng)
                    )
                    if should_stop:
                        game.stop()
                    else:
                        game.continue_turn()
                    decisions += 1
            completed_games += 1
            if completed_games % args.log_every == 0:
                elapsed = time.perf_counter() - started
                print(f"games={completed_games} states={collected} elapsed={elapsed:.1f}s")

    metadata = {
        "dataset": str(args.output),
        "games_requested": args.games,
        "games_collected": completed_games,
        "states": collected,
        "seed": args.seed,
        "rollouts_per_action": args.rollouts,
        "sample_probability": args.sample_probability,
        "exploration_probability": args.exploration_probability,
        "max_turns": args.max_turns,
        "max_future_turns": args.max_future_turns,
        "pairing_policy": "ga_constant_weights",
        "rollout_policy": "rule28-smart_for_both_players_after_forced_action",
        "label": "0_continue_if_q_continue_gt_q_stop_else_1_stop",
        "features_stored": "68_domain_prefixes_are_58_raw_and_60_enhanced",
        "python": platform.python_version(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Dataset: {args.output} ({collected} states from {completed_games} games)")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
