"""Run every pair of saved DQN and classical ML agents head to head."""
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
import itertools
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from cant_stop.evaluation import evaluate
from cant_stop.agents import AGENT_NAMES
from evaluate_ml import paired_bootstrap_interval, write_csv


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    path: Path | None


def parse_model_spec(value: str, kind: str) -> ModelSpec:
    """Parse either PATH or NAME=PATH and produce a safe result label."""
    if "=" in value:
        name, path_text = value.split("=", 1)
    else:
        path_text = value
        name = Path(path_text).stem
    if not name or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
        raise argparse.ArgumentTypeError(
            "model names must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    if not path_text:
        raise argparse.ArgumentTypeError("model path cannot be empty")
    return ModelSpec(name=name, kind=kind, path=Path(path_text))


def parse_baseline_spec(name: str) -> ModelSpec:
    """Represent a built-in heuristic agent in the common tournament format."""
    return ModelSpec(name=name, kind="baseline", path=None)


def load_agent(spec: ModelSpec, device: str, mc_rollouts: int = 64):
    if spec.kind == "dqn":
        # Keep PyTorch optional for classical-only tournaments.
        from cant_stop.dqn import DQNAgent
        assert spec.path is not None
        return DQNAgent.from_checkpoint(spec.path, device=device)
    if spec.kind == "classical_ml":
        from cant_stop.classical_ml import ClassicalMLAgent
        assert spec.path is not None
        return ClassicalMLAgent.from_file(spec.path)
    if spec.kind == "ppo":
        from cant_stop.ppo import PPOAgent, load_checkpoint
        assert spec.path is not None
        model, _ = load_checkpoint(spec.path, device=device)
        return PPOAgent(model, device=device)
    from cant_stop.agents import make_agent
    return make_agent(spec.name, mc_rollouts=mc_rollouts)


def spec_metadata(spec: ModelSpec) -> dict:
    metadata = {"name": spec.name, "kind": spec.kind}
    if spec.path is not None:
        metadata["path"] = str(spec.path)
    return metadata


def _configure_parallel_worker() -> None:
    """Avoid nested PyTorch thread pools competing across worker processes."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # It may already be fixed by an embedding host; inference remains valid.
        pass


def _parallel_matchup(first: ModelSpec, second: ModelSpec, *, games: int,
                      seed: int, max_turns: int, bootstrap_samples: int,
                      device: str, mc_rollouts: int, output_dir: Path) -> dict:
    """Evaluate one matchup in a child process and persist its raw records."""
    started = time.monotonic()
    first_agent = load_agent(first, device, mc_rollouts)
    second_agent = load_agent(second, device, mc_rollouts)
    report = evaluate(
        first_agent, second_agent, games=games, seed=seed,
        max_total_turns=max_turns, include_game_records=True,
    )
    records = report.pop("game_records")
    interval = paired_bootstrap_interval(
        records, samples=bootstrap_samples,
        seed=seed ^ sum(ord(char) for char in first.name + second.name),
    )
    first_rate = report["agents"][0]["win_rate"]
    second_rate = report["agents"][1]["win_rate"]
    report["agents"][0].update({
        "name": first.name,
        "model_kind": first.kind,
        "paired_bootstrap_win_rate95": interval,
    })
    report["agents"][1].update({
        "name": second.name,
        "model_kind": second.kind,
    })
    if first.path is not None:
        report["agents"][0]["model_file"] = str(first.path)
    if second.path is not None:
        report["agents"][1]["model_file"] = str(second.path)
    report["bootstrap_samples"] = bootstrap_samples
    key = f"{first.name}_vs_{second.name}"
    write_csv(output_dir / f"{key}_games.csv", records)
    return {
        "key": key,
        "first": first.name,
        "second": second.name,
        "first_rate": first_rate,
        "second_rate": second_rate,
        "interval": interval,
        "report": report,
        "elapsed_seconds": time.monotonic() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="*", default=[], metavar="[NAME=]PATH",
        help="Classical ML joblib bundles (kept for backwards compatibility)",
    )
    parser.add_argument(
        "--dqn-models", nargs="*", default=[], metavar="[NAME=]PATH",
        help="DQN/Double-DQN checkpoints",
    )
    parser.add_argument(
        "--ppo-models", nargs="*", default=[], metavar="[NAME=]PATH",
        help="PPO checkpoints",
    )
    parser.add_argument(
        "--agents", nargs="*", default=[],
        choices=AGENT_NAMES,
        help="Built-in heuristic agents to include in the tournament",
    )
    parser.add_argument(
        "--mc-rollouts", type=int, default=64,
        help="Rollouts per decision for the built-in mc agent (default: 64)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"),
                        default="cpu", help="Device used for every DQN model")
    parser.add_argument("--games", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=50_000_641)
    parser.add_argument("--max-turns", type=int, default=2_000)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument(
        "--progress-every", type=int, default=50, metavar="GAMES",
        help="Refresh matchup progress every N games; 0 disables it (default: 50)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel matchup processes; 1 runs sequentially (default: 1)",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/classical_ml/head_to_head"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.games, args.max_turns, args.bootstrap_samples, args.mc_rollouts) < 1:
        raise SystemExit("game, turn, bootstrap, and MC rollout counts must be positive")
    if args.progress_every < 0:
        raise SystemExit("progress-every must be >= 0")
    if args.workers < 1:
        raise SystemExit("workers must be >= 1")
    if args.games % 2:
        raise SystemExit("games must be even so every seed has both seat assignments")
    try:
        specs = (
            [parse_model_spec(value, "classical_ml") for value in args.models]
            + [parse_model_spec(value, "dqn") for value in args.dqn_models]
            + [parse_model_spec(value, "ppo") for value in args.ppo_models]
            + [parse_baseline_spec(value) for value in args.agents]
        )
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    if len(specs) < 2:
        raise SystemExit("provide at least two entries across --models, --dqn-models, and --agents")
    missing = [
        str(spec.path) for spec in specs
        if spec.path is not None and not spec.path.is_file()
    ]
    if missing:
        raise SystemExit(f"Model files not found: {', '.join(missing)}")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise SystemExit(
            "model names must be unique; use NAME=PATH to distinguish repeated filenames"
        )

    specs_by_name = {spec.name: spec for spec in specs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, dict[str, float | None]] = {
        name: {opponent: (None if name == opponent else 0.0) for opponent in names}
        for name in names
    }
    matchups: dict[str, dict] = {}

    print("model A                  model B                  A win rate paired bootstrap 95%")
    print("------------------------ ------------------------ ---------- --------------------")
    matchup_count = len(names) * (len(names) - 1) // 2
    matchup_specs = list(itertools.combinations(specs, 2))

    if args.workers > 1:
        worker_count = min(args.workers, matchup_count)
        print(
            f"Running {matchup_count} matchups with {worker_count} worker processes. "
            "Progress is reported when each matchup finishes.",
            flush=True,
        )
        results_by_key: dict[str, dict] = {}
        with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_configure_parallel_worker) as executor:
            futures = {
                executor.submit(
                    _parallel_matchup, first, second, games=args.games,
                    seed=args.seed, max_turns=args.max_turns,
                    bootstrap_samples=args.bootstrap_samples,
                    device=args.device, mc_rollouts=args.mc_rollouts,
                    output_dir=args.output_dir,
                ): (first.name, second.name)
                for first, second in matchup_specs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                first, second = futures[future]
                result = future.result()
                results_by_key[result["key"]] = result
                print(
                    f"Completed {completed}/{matchup_count}: {first} vs {second} "
                    f"({result['elapsed_seconds'] / 60:.1f} min)",
                    flush=True,
                )
        ordered_results = [
            results_by_key[f"{first.name}_vs_{second.name}"]
            for first, second in matchup_specs
        ]
        for result in ordered_results:
            first, second = result["first"], result["second"]
            matchups[result["key"]] = result["report"]
            matrix[first][second] = result["first_rate"]
            matrix[second][first] = result["second_rate"]
            interval = result["interval"]
            print(
                f"{first:<24} {second:<24} {result['first_rate']:>9.1%} "
                f"{interval[0]:>8.1%}-{interval[1]:<8.1%}"
            )
    else:
        agents = {}
        for index, spec in enumerate(specs, start=1):
            print(
                f"Loading model {index}/{len(specs)}: {spec.name} "
                f"({spec.kind}, {spec.path})",
                flush=True,
            )
            agents[spec.name] = load_agent(spec, args.device, args.mc_rollouts)

        for matchup_index, (first_spec, second_spec) in enumerate(
                matchup_specs, start=1):
            first, second = first_spec.name, second_spec.name
            started = time.monotonic()
            print(
                f"Matchup {matchup_index}/{matchup_count}: {first} vs {second}",
                flush=True,
            )

            def show_progress(completed: int, total: int) -> None:
                if args.progress_every == 0:
                    return
                if completed % args.progress_every != 0 and completed != total:
                    return
                elapsed = time.monotonic() - started
                games_per_second = completed / elapsed if elapsed > 0.0 else 0.0
                remaining = ((total - completed) / games_per_second
                             if games_per_second > 0.0 else 0.0)
                print(
                    f"\r  {completed}/{total} ({completed / total:.1%}) | "
                    f"{games_per_second:.2f} games/s | ETA {remaining / 60:.1f} min",
                    end="", flush=True,
                )

            report = evaluate(
                agents[first], agents[second], games=args.games, seed=args.seed,
                max_total_turns=args.max_turns, include_game_records=True,
                progress_callback=show_progress,
            )
            if args.progress_every:
                print()
            records = report.pop("game_records")
            interval = paired_bootstrap_interval(
                records, samples=args.bootstrap_samples,
                seed=args.seed ^ sum(ord(char) for char in first + second),
            )
            first_rate = report["agents"][0]["win_rate"]
            second_rate = report["agents"][1]["win_rate"]
            report["agents"][0]["name"] = first
            report["agents"][1]["name"] = second
            report["agents"][0]["model_kind"] = specs_by_name[first].kind
            report["agents"][1]["model_kind"] = specs_by_name[second].kind
            if specs_by_name[first].path is not None:
                report["agents"][0]["model_file"] = str(specs_by_name[first].path)
            if specs_by_name[second].path is not None:
                report["agents"][1]["model_file"] = str(specs_by_name[second].path)
            report["agents"][0]["paired_bootstrap_win_rate95"] = interval
            report["bootstrap_samples"] = args.bootstrap_samples
            key = f"{first}_vs_{second}"
            matchups[key] = report
            matrix[first][second] = first_rate
            matrix[second][first] = second_rate
            write_csv(args.output_dir / f"{key}_games.csv", records)
            print(
                f"{first:<24} {second:<24} {first_rate:>9.1%} "
                f"{interval[0]:>8.1%}-{interval[1]:<8.1%}"
            )

    matrix_path = args.output_dir / "win_rate_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *names])
        for name in names:
            writer.writerow([name, *[
                "" if matrix[name][opponent] is None else matrix[name][opponent]
                for opponent in names
            ]])

    summary_path = args.output_dir / "head_to_head_summary.json"
    summary_path.write_text(json.dumps({
        "games_per_matchup": args.games,
        "seed": args.seed,
        "device": args.device,
        "workers": min(args.workers, matchup_count),
        "mc_rollouts": args.mc_rollouts,
        "pairing_policy": (
            "agent-specific; couto uses its original secured-progress pairing, "
            "while couto-ga-stop and training-facing learned agents use fixed GA pairing"
        ),
        "paired_bootstrap_unit": "shared_dice_seed_two_seat_games",
        "models": [spec_metadata(spec) for spec in specs],
        "matrix_rows_model_win_rate_vs_columns": matrix,
        "matchups": matchups,
    }, indent=2), encoding="utf-8")
    print(f"Matrix: {matrix_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
