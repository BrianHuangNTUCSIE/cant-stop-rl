"""Run a reproducible, metadata-rich matrix of registered agents.

This runner performs evaluation only.  It does not train models or create
synthetic data.  Each matchup writes a JSON summary and, when requested, the
raw per-game records needed for later paired analysis.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cant_stop.agents import AGENT_NAMES, make_agent
from cant_stop.evaluation import evaluate


def search_budget(agent_name: str, rollouts: int) -> dict:
    """Describe the actual budget rather than calling every search MCTS."""
    if agent_name == "vmerckle_monte":
        return {
            "rollouts": rollouts,
            "allocation": "floor(rollouts / root_actions) per root action",
            "tree": "none",
        }
    if agent_name == "vmerckle_monte_uct":
        return {
            "rollouts": rollouts,
            "allocation": "one initial sample per root action, then UCB1",
            "tree": "root-only bandit",
        }
    if agent_name in {"mc", "mc-strong"}:
        return {
            "rollouts": rollouts,
            "allocation": "remainder-of-turn samples",
            "tree": "none",
        }
    if agent_name == "mc-full-game":
        return {
            "rollouts": rollouts,
            "allocation": "paired stop/continue full-game samples",
            "tree": "none",
        }
    return {"rollouts": 0, "allocation": "deterministic or stochastic heuristic", "tree": "none"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents", default="glenn2009_constant,glenn2010_linear,rule28-smart,mc",
        help="Comma-separated registered agents (default: core comparison set)",
    )
    parser.add_argument("--opponent", default="rule28-smart", choices=AGENT_NAMES)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=441000)
    parser.add_argument("--replicates", type=int, default=1,
                        help="Independent evaluation batches per matchup")
    parser.add_argument("--seed-stride", type=int, default=1_000_000,
                        help="Seed spacing between matchup/replicate batches")
    parser.add_argument("--mc-rollouts", type=int, default=32)
    parser.add_argument("--mc-future-roll-limit", type=int, default=40)
    parser.add_argument("--max-turns", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-raw", action="store_true",
                        help="Do not include per-game records in JSON output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    names = tuple(dict.fromkeys(name.strip() for name in args.agents.split(",") if name.strip()))
    if not names:
        raise SystemExit("--agents must contain at least one agent")
    if (args.games < 1 or args.replicates < 1 or args.seed_stride < 1
            or args.mc_rollouts < 1 or args.mc_future_roll_limit < 1):
        raise SystemExit("games, replicates, seed-stride, and rollout limits must be positive")
    if args.max_turns < 1:
        raise SystemExit("max-turns must be positive")
    if args.seed_stride < len(names):
        raise SystemExit("seed-stride must be at least the number of matchups")
    unknown = [name for name in names if name not in AGENT_NAMES]
    if unknown:
        raise SystemExit(f"Unknown agent(s): {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "runner": "scripts/evaluate_agent_comparison.py",
        "agents": names,
        "opponent": args.opponent,
        "games_per_matchup": args.games,
        "base_seed": args.seed,
        "replicates": args.replicates,
        "seed_stride": args.seed_stride,
        "seed_formula": (
            "seed = base_seed + matchup_index + replicate_index * seed_stride"
        ),
        "mc_rollouts": args.mc_rollouts,
        "mc_future_roll_limit": args.mc_future_roll_limit,
        "max_total_turns": args.max_turns,
        "python": sys.version,
        "platform": platform.platform(),
        "raw_records": not args.no_raw,
        "inference_note": (
            "Evaluator alternates seats and reuses each pair seed for the two "
            "seat assignments; use paired bootstrap or between-seed variability."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    for matchup_index, name in enumerate(names):
        for replicate_index in range(args.replicates):
            evaluation_seed = args.seed + matchup_index + replicate_index * args.seed_stride
            started = time.perf_counter()
            report = evaluate(
                make_agent(name, args.mc_rollouts, args.mc_future_roll_limit),
                make_agent(args.opponent, args.mc_rollouts, args.mc_future_roll_limit),
                games=args.games,
                seed=evaluation_seed,
                max_total_turns=args.max_turns,
                include_game_records=not args.no_raw,
            )
            elapsed_seconds = time.perf_counter() - started
            report["metadata"] = {
                **manifest,
                "matchup_index": matchup_index,
                "replicate_index": replicate_index,
                "agent_a": name,
                "agent_b": args.opponent,
                "evaluation_seed": evaluation_seed,
                "elapsed_seconds": elapsed_seconds,
                "search_budget": search_budget(name, args.mc_rollouts),
                "opponent_search_budget": search_budget(args.opponent, args.mc_rollouts),
            }
            suffix = f"_r{replicate_index:02d}" if args.replicates > 1 else ""
            path = args.output_dir / f"{matchup_index:02d}{suffix}_{name}_vs_{args.opponent}.json"
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"{name} [replicate {replicate_index}] vs {args.opponent}: "
                  f"{report['agents'][0]['win_rate']:.1%} "
                  f"({report['agents'][0]['wins']}/{args.games}) -> {path}")


if __name__ == "__main__":
    main()
