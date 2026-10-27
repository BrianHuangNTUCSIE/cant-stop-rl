"""Directly compare the original MC baseline with full-game ``mc-strong``."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cant_stop.agents import make_agent
from cant_stop.evaluation import evaluate
from evaluate_ml import paired_bootstrap_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30_000_641)
    parser.add_argument("--max-turns", type=int, default=2_000)
    parser.add_argument("--mc-rollouts", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/mc_vs_mc_strong"))
    args = parser.parse_args()
    if min(args.games, args.max_turns, args.mc_rollouts,
           args.bootstrap_samples) < 1:
        raise SystemExit("games, max-turns, rollouts, and bootstrap-samples must be positive")
    if args.games % 2:
        raise SystemExit("games must be even so each seed gets both seat assignments")
    if args.progress_every < 1:
        raise SystemExit("progress-every must be positive")

    original = make_agent("mc", args.mc_rollouts)
    strong = make_agent("mc-strong", args.mc_rollouts)

    def progress(completed: int, total: int) -> None:
        if completed % args.progress_every == 0 or completed == total:
            print(f"\r{completed}/{total} ({completed / total:.1%})",
                  end="", flush=True)

    report = evaluate(
        original, strong, games=args.games, seed=args.seed,
        max_total_turns=args.max_turns, include_game_records=True,
        progress_callback=progress,
    )
    print()
    records = report.pop("game_records")
    interval = paired_bootstrap_interval(
        records, samples=args.bootstrap_samples,
        seed=args.seed ^ 0x4D435354,
    )
    report["agents"][0]["name"] = "mc"
    report["agents"][1]["name"] = "mc-strong"
    report["agents"][0]["paired_bootstrap_win_rate95"] = interval
    report["mc_rollouts"] = args.mc_rollouts
    report["bootstrap_samples"] = args.bootstrap_samples

    args.output_dir.mkdir(parents=True, exist_ok=True)
    games_path = args.output_dir / "mc_vs_mc_strong_games.csv"
    with games_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    first, second = report["agents"]
    print(f"mc win rate:        {first['win_rate']:.1%}")
    print(f"mc-strong win rate: {second['win_rate']:.1%}")
    print(f"mc paired 95% CI:   {interval[0]:.1%}-{interval[1]:.1%}")
    print(f"Games: {games_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
