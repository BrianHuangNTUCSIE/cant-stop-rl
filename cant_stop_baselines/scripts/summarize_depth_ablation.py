"""Aggregate the controlled DDQN depth-ablation experiment."""
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
from pathlib import Path


ARCHITECTURES = (
    ("control_128x128", (128, 128)),
    ("wide_256x256", (256, 256)),
    ("deep_256x256x128", (256, 256, 128)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    for name, hidden_sizes in ARCHITECTURES:
        architecture_dir = args.root / name
        summary_path = architecture_dir / "multi_seed_summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        statuses = []
        for run in summary["runs"]:
            status_path = Path(run["output_dir"]) / "training_status.json"
            if status_path.is_file():
                statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
        row: dict[str, object] = {
            "architecture": name,
            "hidden_sizes": ",".join(map(str, hidden_sizes)),
            "successful_seeds": summary["successful_runs"],
            "failed_seeds": len(summary["failures"]),
            "early_stopped_seeds": sum(
                bool(status["early_stopped"]) for status in statuses
            ),
            "mean_selection_score": summary["mean_selection_score"],
        }
        for opponent, aggregate in summary["aggregate_by_opponent"].items():
            row[f"{opponent}_mean_win_rate"] = aggregate["mean_win_rate"]
            row[f"{opponent}_sample_stddev"] = aggregate["sample_stddev"]
        rows.append(row)
        details[name] = {
            "hidden_sizes": hidden_sizes,
            "multi_seed_summary": summary,
            "training_statuses": statuses,
        }

    if not rows:
        raise SystemExit(f"No completed architecture summaries found under {args.root}")

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (args.root / "depth_ablation_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "depth_ablation_summary.json").write_text(
        json.dumps({
            "controlled_variable": "hidden_sizes",
            "architectures": details,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Saved depth-ablation summary to {args.root}")


if __name__ == "__main__":
    main()
