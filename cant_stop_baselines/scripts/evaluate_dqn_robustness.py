"""Evaluate saved DQN checkpoints against a common robustness opponent set."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cant_stop_core.agents import make_agent
from cant_stop_core.classical_ml import ClassicalMLAgent
from cant_stop_core.dqn import DQNAgent
from cant_stop_core.evaluation import evaluate
from evaluate_ml import paired_bootstrap_interval


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path_text = value.split("=", 1)
    if not name or not path_text:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path_text)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dqn", action="append", required=True, type=parse_named_path,
                        metavar="NAME=CHECKPOINT")
    parser.add_argument("--ridge", type=Path, required=True)
    parser.add_argument("--opponents", nargs="+",
                        default=["rule28-smart", "mc", "random", "rule28"])
    parser.add_argument("--games", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=60_0447)
    parser.add_argument("--max-turns", type=int, default=2_000)
    parser.add_argument("--mc-rollouts", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--device", default="cpu",
                        choices=("auto", "cpu", "cuda", "mps", "xpu"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.games < 2 or args.games % 2:
        raise SystemExit("games must be an even number >= 2")
    if not args.ridge.is_file():
        raise SystemExit(f"Ridge model not found: {args.ridge}")
    for name, path in args.dqn:
        if not path.is_file():
            raise SystemExit(f"DQN checkpoint not found: {path}")
        if name == "ridge" or name in args.opponents:
            raise SystemExit(f"duplicate model/opponent name: {name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = [(name, DQNAgent.from_checkpoint(path, device=args.device), str(path))
              for name, path in args.dqn]
    opponents = []
    for name in args.opponents:
        opponents.append((name, make_agent(name, mc_rollouts=args.mc_rollouts), None))
    opponents.append(("ridge", ClassicalMLAgent.from_file(args.ridge), str(args.ridge)))

    summary: dict[str, object] = {
        "games_per_matchup": args.games,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "mc_rollouts": args.mc_rollouts,
        "bootstrap_samples": args.bootstrap_samples,
        "models": [{"name": name, "checkpoint": path} for name, _, path in models],
        "opponents": [name for name, _, _ in opponents],
        "matchups": {},
    }

    total = len(models) * len(opponents)
    completed = 0
    for model_name, model, model_path in models:
        for opponent_name, opponent, opponent_path in opponents:
            completed += 1
            print(f"[{completed}/{total}] {model_name} vs {opponent_name}", flush=True)
            started = time.monotonic()
            report = evaluate(
                model, opponent, games=args.games, seed=args.seed,
                max_total_turns=args.max_turns, include_game_records=True,
            )
            records = report.pop("game_records")
            interval = paired_bootstrap_interval(
                records, samples=args.bootstrap_samples,
                seed=args.seed ^ sum(ord(c) for c in model_name + opponent_name),
            )
            report["agents"][0]["name"] = model_name
            report["agents"][0]["checkpoint"] = model_path
            report["agents"][0]["paired_bootstrap_win_rate95"] = interval
            report["agents"][1]["name"] = opponent_name
            report["agents"][1]["model_file"] = opponent_path
            report["bootstrap_samples"] = args.bootstrap_samples
            key = f"{model_name}_vs_{opponent_name}"
            write_csv(args.output_dir / f"{key}_games.csv", records)
            summary["matchups"][key] = report
            rate = report["agents"][0]["win_rate"]
            print(f"    win rate={rate:.1%}, elapsed={(time.monotonic() - started) / 60:.1f} min",
                  flush=True)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved robustness results to {args.output_dir}")


if __name__ == "__main__":
    main()
