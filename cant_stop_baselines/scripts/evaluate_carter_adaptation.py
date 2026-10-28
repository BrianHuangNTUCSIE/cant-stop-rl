"""Reproduce the public Carter checkpoint adaptation calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cant_stop.carter_adapter import CarterNumpyAgent
from cant_stop.evaluation import evaluate
from cant_stop.agents import RandomAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--max-turns", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 1 or args.max_turns < 1:
        parser.error("games and max-turns must be positive")
    report = evaluate(
        CarterNumpyAgent(args.checkpoint), RandomAgent(), games=args.games,
        seed=args.seed, max_total_turns=args.max_turns, include_game_records=True,
    )
    report["metadata"] = {
        "runner": "scripts/evaluate_carter_adaptation.py",
        "agent_a": "carter-public-adapted",
        "agent_b": "random",
        "games_per_matchup": args.games,
        "evaluation_seed": args.seed,
        "max_total_turns": args.max_turns,
        "pairing_policy": "Carter heuristic selector",
        "feature_mode": "Carter-compatible static Prob_Table",
        "checkpoint_source": "https://github.com/JCameronCarter/HonorsThesis/blob/main/Code/dqn_multi_opp.keras",
        "checkpoint_git_blob_sha": "e225a7571b1b70d2168272b21e6fed7d70c10a85",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_architecture": "51 -> 64 -> 64 -> 2",
        "inference": "equivalent NumPy Dense/ReLU network loaded from Keras weights; no retraining",
        "adaptation": "current simulator rules and Carter-compatible feature encoder",
        "python": sys.version,
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
