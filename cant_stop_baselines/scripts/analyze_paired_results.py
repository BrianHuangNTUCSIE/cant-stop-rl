"""Analyze JSON results from ``evaluate_agent_comparison.py``.

The analysis uses dice-seed pairs rather than treating seat-swapped games as
independent observations.  It performs no game simulation or model training.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cant_stop_core.paired_analysis import paired_bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON matchup report or directory of reports")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=441)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        # Matrix reports have this stable suffix; ignore manifests and prior
        # analysis outputs that may live beside them.
        return sorted(input_path.glob("*_vs_*.json"))
    return [input_path]


def main() -> None:
    args = build_parser().parse_args()
    if args.resamples < 1:
        raise SystemExit("--resamples must be positive")
    reports = []
    for path in _paths(args.input):
        report = json.loads(path.read_text(encoding="utf-8"))
        if "game_records" not in report:
            raise SystemExit(f"{path} has no game_records; rerun evaluation without --no-raw")
        result = paired_bootstrap(report["game_records"], args.resamples, args.seed)
        metadata = report.get("metadata", {})
        result["input"] = str(path)
        result["agent_a"] = metadata.get("agent_a")
        result["agent_b"] = metadata.get("agent_b")
        result["evaluation_seed"] = metadata.get("evaluation_seed", report.get("seed"))
        reports.append(result)
    if len(reports) == 1:
        payload: object = reports[0]
    else:
        groups: dict[tuple[object, object], list[dict]] = defaultdict(list)
        for result in reports:
            groups[(result.get("agent_a"), result.get("agent_b"))].append(result)
        summaries = []
        for (agent_a, agent_b), group in groups.items():
            margins = [float(item["observed_margin_a_minus_b"]) for item in group]
            mean = sum(margins) / len(margins)
            variance = (sum((margin - mean) ** 2 for margin in margins)
                        / (len(margins) - 1) if len(margins) > 1 else 0.0)
            summaries.append({
                "agent_a": agent_a,
                "agent_b": agent_b,
                "replicates": len(group),
                "mean_margin_a_minus_b": mean,
                "sample_sd_margin": math.sqrt(variance),
                "min_margin": min(margins),
                "max_margin": max(margins),
                "interpretation": "between-seed descriptive variability; no significance claim",
            })
        payload = {"results": reports, "between_seed_summary": summaries}
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
