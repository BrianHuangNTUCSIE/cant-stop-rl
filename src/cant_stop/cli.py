"""Command-line interface for baseline evaluation."""
import argparse
import csv
import json
from pathlib import Path

from .agents import AGENT_NAMES, make_agent
from .evaluation import evaluate
from .sketch_program import SketchProgramAgent
from .carter_adapter import CarterKerasAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Can't Stop baseline evaluation (2 players)")
    parser.add_argument("--agent-a", default="rule28-smart",
                        choices=AGENT_NAMES)
    parser.add_argument("--agent-b", default="random",
                        choices=AGENT_NAMES)
    parser.add_argument("--sketch-program-a", type=str, default=None,
                        help="Load a version-1 JSON sketch for player A")
    parser.add_argument("--sketch-program-b", type=str, default=None,
                        help="Load a version-1 JSON sketch for player B")
    parser.add_argument("--carter-checkpoint-a", type=str, default=None,
                        help="Load a local Carter 51-feature Keras checkpoint for player A")
    parser.add_argument("--carter-checkpoint-b", type=str, default=None,
                        help="Load a local Carter 51-feature Keras checkpoint for player B")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc-rollouts", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=2000)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write per-game metrics to this CSV path")
    args = parser.parse_args()
    if sum(value is not None for value in
           (args.sketch_program_a, args.carter_checkpoint_a)) > 1:
        raise SystemExit("Choose only one custom agent source for player A")
    if sum(value is not None for value in
           (args.sketch_program_b, args.carter_checkpoint_b)) > 1:
        raise SystemExit("Choose only one custom agent source for player B")
    agent_a = (SketchProgramAgent.from_json(args.sketch_program_a)
               if args.sketch_program_a else
               CarterKerasAgent(args.carter_checkpoint_a)
               if args.carter_checkpoint_a else make_agent(args.agent_a, args.mc_rollouts))
    agent_b = (SketchProgramAgent.from_json(args.sketch_program_b)
               if args.sketch_program_b else
               CarterKerasAgent(args.carter_checkpoint_b)
               if args.carter_checkpoint_b else make_agent(args.agent_b, args.mc_rollouts))
    result = evaluate(agent_a, agent_b, args.games,
                      args.seed, args.max_turns, include_game_records=args.csv is not None)
    if args.csv:
        if not result.get("game_records"):
            raise SystemExit("CSV output requires at least one game record")
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=result["game_records"][0].keys())
            writer.writeheader()
            writer.writerows(result["game_records"])
    result["agents"][0]["name"] = (
        f"sketch:{args.sketch_program_a}" if args.sketch_program_a else
        f"carter:{args.carter_checkpoint_a}" if args.carter_checkpoint_a else args.agent_a
    )
    result["agents"][1]["name"] = (
        f"sketch:{args.sketch_program_b}" if args.sketch_program_b else
        f"carter:{args.carter_checkpoint_b}" if args.carter_checkpoint_b else args.agent_b
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Games: {result['games']}   Draws: {result['draws']}   Seed: {args.seed}")
        for x in result["agents"]:
            ci = x["win_rate_wilson95"]
            print(f"{x['id']} ({x['name']}): wins={x['wins']} "
                  f"win_rate={x['win_rate']:.1%} "
                  f"95% CI=[{ci[0]:.1%}, {ci[1]:.1%}] "
                  f"avg_turns={x['avg_turns_per_game']:.2f} "
                  f"bust/turn={x['busts_per_turn']:.1%}")
