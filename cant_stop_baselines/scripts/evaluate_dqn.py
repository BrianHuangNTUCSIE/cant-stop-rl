"""Evaluate a saved DQN checkpoint against a baseline agent."""
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
import time
from pathlib import Path

from cant_stop_core.agents import AGENT_NAMES, RandomAgent, make_agent
from cant_stop_core.dqn import DQNAgent
from cant_stop_core.evaluation import evaluate


OPPONENTS = AGENT_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent", choices=("all",) + OPPONENTS, default="all",
                        help="Baseline opponent (default: evaluate all baselines)")
    parser.add_argument("--games", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=30_000_441)
    parser.add_argument("--max-turns", type=int, default=2_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"),
                        default="cpu")
    parser.add_argument("--random-stop-probability", type=float, default=0.3)
    parser.add_argument(
        "--mc-rollouts", type=int, default=64,
        help="Rollouts per MC decision (default: 64; matches head-to-head evaluation)",
    )
    parser.add_argument("--summary", type=Path, default=None,
                        help="Optionally save summary JSON (single opponent only)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optionally save per-game CSV (single opponent only)")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="Refresh progress every N games; 0 disables it")
    return parser


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.games < 1:
        raise SystemExit("games must be >= 1")
    if args.max_turns < 1:
        raise SystemExit("max-turns must be >= 1")
    if not 0.0 <= args.random_stop_probability <= 1.0:
        raise SystemExit("random-stop-probability must be between 0 and 1")
    if args.mc_rollouts < 1:
        raise SystemExit("mc-rollouts must be >= 1")
    if args.progress_every < 0:
        raise SystemExit("progress-every must be >= 0")

    if args.opponent == "all" and (args.summary is not None or args.csv is not None):
        raise SystemExit("--summary and --csv require one explicit --opponent")

    learned = DQNAgent.from_checkpoint(args.checkpoint, device=args.device)
    selected_opponents = OPPONENTS if args.opponent == "all" else (args.opponent,)
    table_rows: list[tuple[str, int, int, float, float, float, int]] = []
    for opponent_key in selected_opponents:
        if opponent_key == "random":
            opponent = RandomAgent(stop_probability=args.random_stop_probability)
            opponent_name = f"random-p{args.random_stop_probability:g}"
        else:
            opponent = make_agent(opponent_key, args.mc_rollouts)
            opponent_name = opponent_key

        started = time.monotonic()

        def show_progress(completed: int, total: int) -> None:
            if args.progress_every == 0:
                return
            if completed % args.progress_every != 0 and completed != total:
                return
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed else 0.0
            remaining = (total - completed) / rate if rate else 0.0
            print(
                f"\r[evaluate:{opponent_name}] {completed}/{total} "
                f"({completed / total:.1%}) | {rate:.2f} games/s | "
                f"ETA {remaining / 60:.1f} min",
                end="", flush=True,
            )

        report = evaluate(
            learned,
            opponent,
            games=args.games,
            seed=args.seed,
            max_total_turns=args.max_turns,
            include_game_records=args.csv is not None,
            progress_callback=show_progress,
        )
        if args.progress_every:
            print()
        records = report.pop("game_records", None)
        report["checkpoint"] = str(args.checkpoint)
        report["agents"][0]["name"] = f"learned-{args.checkpoint.stem}"
        report["agents"][1]["name"] = opponent_name

        if args.summary is not None:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.csv is not None:
            assert records is not None
            write_csv(args.csv, records)

        learned_result, opponent_result = report["agents"]
        low, high = learned_result["win_rate_wilson95"]
        table_rows.append((
            opponent_name,
            learned_result["wins"],
            opponent_result["wins"],
            learned_result["win_rate"],
            low,
            high,
            report["draws"],
        ))

    print(f"Checkpoint: {args.checkpoint}  Games/opponent: {args.games}  Seed: {args.seed}")
    print("+----------------+------------+----------+----------+-----------------+-------+")
    print("| Opponent       | Agent wins | Opp wins | Win rate | Agent 95% CI    | Draws |")
    print("+----------------+------------+----------+----------+-----------------+-------+")
    for name, agent_wins, opponent_wins, rate, low, high, draws in table_rows:
        interval = f"{low:.1%}-{high:.1%}"
        print(
            f"| {name:<14} | {agent_wins:>10} | {opponent_wins:>8} | "
            f"{rate:>8.1%} | {interval:>15} | {draws:>5} |"
        )
    print("+----------------+------------+----------+----------+-----------------+-------+")
    if args.summary is not None:
        print(f"Summary: {args.summary}")
    if args.csv is not None:
        print(f"Games CSV: {args.csv}")


if __name__ == "__main__":
    main()
