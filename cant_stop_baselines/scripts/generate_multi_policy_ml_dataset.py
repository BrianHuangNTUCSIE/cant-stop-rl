"""Collect states from several policies and label all of them identically."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cant_stop_core.agents import make_agent
from cant_stop_core.game import Game
from cant_stop_core.model import State
from cant_stop_core.rl_env import StopContinueAction
from evaluate_ml import ModelSpec, load_saved_agent
from generate_ml_dataset import label_state

FEATURE_COLUMNS = [f"x{i:02d}" for i in range(68)]


def parse_source(value: str) -> tuple[str, str | None]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, path
    return value, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True,
                        help="builtin names or NAME=checkpoint paths")
    parser.add_argument("--games-per-source", type=int, default=500)
    parser.add_argument("--max-states", type=int, default=10000)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--max-turns", type=int, default=2000)
    parser.add_argument("--max-future-turns", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [parse_source(value) for value in args.sources]
    agents = []
    for name, path in sources:
        if path is None:
            agents.append(make_agent(name, args.rollouts))
        else:
            agents.append(load_saved_agent(ModelSpec(name, Path(path)), "cpu")[0])
    fields = ["source", "game_id", "game_seed", "decision_index", "player",
              "rolls_this_turn", "rollouts", "q_stop", "q_continue",
              "continue_margin", "label", *FEATURE_COLUMNS]
    collected = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source_index, ((source_name, _), agent) in enumerate(zip(sources, agents)):
            for local_game in range(args.games_per_source):
                if collected >= args.max_states:
                    break
                game_id = source_index * args.games_per_source + local_game
                game_seed = args.seed + game_id * 1009
                dice_rng = random.Random(game_seed)
                agent_rngs = (random.Random(game_seed ^ 0xA11CE001),
                              random.Random(game_seed ^ 0xA11CE002))
                sampling_rng = random.Random(game_seed ^ 0x51A7E000)
                game = Game(); decisions = 0
                while game.winner is None and sum(game.turns) < args.max_turns:
                    player = game.current_player
                    if game.phase == "roll":
                        if not game.roll(dice_rng):
                            continue
                    if game.phase == "pair":
                        state = game.snapshot()
                        game.apply_move(agent.choose_move(state, game.available,
                                                          agent_rngs[player]))
                    if game.phase == "decision":
                        state = game.snapshot()
                        if sampling_rng.random() < 0.5 and collected < args.max_states:
                            label_seed = (args.seed + 1_000_000_000 + game_id * 1_000_003
                                          + decisions * 10_007)
                            q_stop, q_continue = label_state(
                                state, rollouts=args.rollouts, seed=label_seed,
                                max_future_turns=args.max_future_turns)
                            margin = q_continue - q_stop
                            row = {"source": source_name, "game_id": game_id,
                                   "game_seed": game_seed, "decision_index": decisions,
                                   "player": player, "rolls_this_turn": state.rolls_this_turn,
                                   "rollouts": args.rollouts, "q_stop": q_stop,
                                   "q_continue": q_continue, "continue_margin": margin,
                                   "label": int(StopContinueAction.CONTINUE if margin > 0 else StopContinueAction.STOP)}
                            row.update(zip(FEATURE_COLUMNS, state.features(domain=True)))
                            writer.writerow(row); collected += 1
                        if agent.should_stop(state, agent_rngs[player]):
                            game.stop()
                        else:
                            game.continue_turn()
                        decisions += 1
            print(f"source={source_name} states={collected}")
            if collected >= args.max_states:
                break
    print(f"Dataset: {args.output} states={collected}")


if __name__ == "__main__":
    main()
