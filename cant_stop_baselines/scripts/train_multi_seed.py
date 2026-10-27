"""Run identical DQN experiments for several seeds and aggregate held-out results."""
from __future__ import annotations

import sys
from pathlib import Path

# Extend the import path to the project root so entry points work when
# launched as ``python scripts/<name>.py`` from the repository root.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("provide one or more unique seeds")
    return seeds


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog=(
            "Unknown arguments are forwarded to train_dqn.py. Example: "
            "--seeds 441,442,443 --episodes 100000 --domain-features --n-step 5"
        ),
    )
    parser.add_argument("--seeds", type=parse_seeds, default=(441, 442, 443, 444, 445))
    parser.add_argument("--output-root", type=Path, default=Path("runs/multi_seed"))
    parser.add_argument(
        "--jobs", type=positive_int, default=1,
        help="Retained for compatibility; must be 1 (seeds are never run in parallel)",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> None:
    args, training_args = build_parser().parse_known_args()
    if args.jobs != 1:
        raise SystemExit(
            "parallel training of different seeds is disabled; use --jobs 1"
        )
    forbidden = ("--seed", "--output-dir")
    if any(arg == name or arg.startswith(name + "=")
           for arg in training_args for name in forbidden):
        raise SystemExit("--seed and --output-dir are managed by train_multi_seed.py")

    script = Path(__file__).with_name("train_dqn.py")
    args.output_root.mkdir(parents=True, exist_ok=True)
    def run_seed(seed: int) -> tuple[int, Path, int]:
        output_dir = args.output_root / f"seed{seed}"
        command = [
            sys.executable, str(script), *training_args,
            "--seed", str(seed), "--output-dir", str(output_dir),
        ]
        print(f"Running seed {seed}: {' '.join(command)}", flush=True)
        if args.jobs == 1:
            result = subprocess.run(command, check=False)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "run.log"
            with log_path.open("w", encoding="utf-8") as log_handle:
                log_handle.write(f"Command: {' '.join(command)}\n")
                log_handle.flush()
                result = subprocess.run(
                    command, check=False, stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
        print(f"Seed {seed} finished with exit code {result.returncode}", flush=True)
        return seed, output_dir, result.returncode

    # Keep seed runs strictly sequential.  Besides being an experiment
    # requirement, this prevents CPU oversubscription from hiding the
    # throughput of one controlled training run.
    outcomes = [run_seed(seed) for seed in args.seeds]

    runs: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for seed, output_dir, returncode in outcomes:
        if returncode != 0:
            failures.append({"seed": seed, "returncode": returncode})
            continue
        summary_path = output_dir / "best_evaluation_summary.json"
        if not summary_path.exists():
            # Runs with interim evaluation disabled have no best checkpoint;
            # aggregate their final checkpoint instead.
            summary_path = Path("results") / output_dir.name / "evaluation_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append({
            "seed": seed,
            "output_dir": str(output_dir),
            "selection_score": summary["selection_score"],
            "win_rates": {
                name: report["agents"][0]["win_rate"]
                for name, report in summary["opponents"].items()
            },
        })

    opponents = sorted({
        name for run in runs for name in run["win_rates"]  # type: ignore[union-attr]
    })
    aggregate: dict[str, dict[str, float]] = {}
    for name in opponents:
        values = [run["win_rates"][name] for run in runs]  # type: ignore[index]
        aggregate[name] = {
            "mean_win_rate": statistics.fmean(values),
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_win_rate": min(values),
            "max_win_rate": max(values),
        }
    scores = [float(run["selection_score"]) for run in runs]
    report = {
        "seeds_requested": args.seeds,
        "successful_runs": len(runs),
        "failures": failures,
        "runs": runs,
        "aggregate_by_opponent": aggregate,
        "mean_selection_score": statistics.fmean(scores) if scores else None,
        "training_arguments": training_args,
    }
    destination = args.output_root / "multi_seed_summary.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved multi-seed summary to {destination}")
    if failures and not args.continue_on_error:
        raise SystemExit(int(failures[0]["returncode"]))


if __name__ == "__main__":
    main()
