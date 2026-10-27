"""Evaluate saved classical, DQN, and PPO agents in complete seeded games."""
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
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from cant_stop.agents import AGENT_NAMES, make_agent
from cant_stop.classical_ml import ClassicalMLAgent
from cant_stop.evaluation import evaluate


OPPONENTS = AGENT_NAMES
PPO_PAIRING_POLICIES = ("ga", "couto")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    ppo_pairing_policy: str | None = None


def parse_model_spec(value: str) -> ModelSpec:
    """Parse PATH or NAME=PATH, allowing duplicate checkpoint filenames."""
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
    return ModelSpec(name, Path(path_text))


def parse_ppo_pairing(value: str) -> tuple[str, str]:
    """Parse a model-specific frozen PPO pairing override."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("PPO pairing overrides use NAME=POLICY")
    name, policy = value.split("=", 1)
    if not name or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
        raise argparse.ArgumentTypeError("PPO pairing override has an invalid model name")
    if policy not in PPO_PAIRING_POLICIES:
        raise argparse.ArgumentTypeError(
            f"PPO pairing policy must be one of {', '.join(PPO_PAIRING_POLICIES)}"
        )
    return name, policy


def load_saved_agent(spec: ModelSpec, device: str):
    """Load a supported model while keeping optional dependencies lazy."""
    if spec.path.suffix not in (".pt", ".pth"):
        if spec.ppo_pairing_policy is not None:
            raise ValueError("PPO pairing overrides can only be used with PPO checkpoints")
        return ClassicalMLAgent.from_file(spec.path), "classical-ml"

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Neural checkpoint evaluation requires PyTorch. Install "
            "requirements-rl.txt with a supported Python version."
        ) from exc
    payload = torch.load(spec.path, map_location="cpu", weights_only=False)
    algorithm = payload.get("algorithm")
    if algorithm is None and isinstance(payload.get("config"), dict):
        algorithm = payload["config"].get("algorithm")
    if algorithm == "supervised-dnn":
        from cant_stop.supervised_dnn import load_checkpoint
        return load_checkpoint(spec.path, device=device), "supervised-dnn"
    if algorithm == "ppo" or "model_state_dict" in payload:
        from cant_stop.ppo import PPOAgent, load_checkpoint
        model, _ = load_checkpoint(spec.path, device=device)
        pairing_policy = None
        if spec.ppo_pairing_policy is not None:
            pairing_policy = make_agent(
                "rule28-smart" if spec.ppo_pairing_policy == "ga" else "couto"
            )
        return PPOAgent(
            model, pairing_policy=pairing_policy, device=device
        ), "ppo"
    if algorithm in ("dqn", "double-dqn") or "online_state_dict" in payload:
        if spec.ppo_pairing_policy is not None:
            raise ValueError("PPO pairing overrides can only be used with PPO checkpoints")
        from cant_stop.dqn import DQNAgent
        return DQNAgent.from_checkpoint(spec.path, device=device), str(algorithm or "dqn")
    raise ValueError(f"Unrecognized neural checkpoint format: {spec.path}")


def _configure_parallel_worker() -> None:
    """Prevent nested PyTorch thread pools from oversubscribing the CPU."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # An embedding host may already have fixed the inter-op thread count.
        pass


def paired_bootstrap_interval(records: list[dict], *, samples: int,
                              seed: int) -> tuple[float, float]:
    """Bootstrap shared-dice seat-swap pairs, preserving pair dependence."""
    by_seed: dict[int, list[float]] = {}
    for row in records:
        by_seed.setdefault(int(row["seed"]), []).append(
            1.0 if row["winner"] == "A" else 0.0
        )
    pairs = [sum(values) / len(values) for values in by_seed.values()]
    if not pairs:
        return (0.0, 1.0)
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(pairs) for _ in pairs) / len(pairs)
        for _ in range(samples)
    )
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return low, high


def scores_by_seed(records: list[dict]) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in records:
        grouped.setdefault(int(row["seed"]), []).append(
            1.0 if row["winner"] == "A" else 0.0
        )
    return {seed: sum(values) / len(values) for seed, values in grouped.items()}


def paired_difference_interval(first: dict[int, float], second: dict[int, float],
                               *, samples: int, seed: int) -> tuple[float, float, float]:
    shared = sorted(first.keys() & second.keys())
    if not shared:
        raise ValueError("Models have no shared evaluation seeds")
    differences = [first[key] - second[key] for key in shared]
    estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrapped = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return (
        estimate,
        bootstrapped[int(0.025 * (samples - 1))],
        bootstrapped[int(0.975 * (samples - 1))],
    )


def parse_opponents(value: str) -> tuple[str, ...]:
    if value == "all":
        return OPPONENTS
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or any(name not in OPPONENTS for name in names):
        raise argparse.ArgumentTypeError(f"opponents must come from {', '.join(OPPONENTS)}")
    return names


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _parallel_matchup(model_spec: ModelSpec, opponent_name: str, *, games: int,
                      seed: int, max_turns: int, mc_rollouts: int,
                      bootstrap_samples: int, device: str,
                      output_dir: Path) -> dict:
    """Evaluate and persist one independent model-opponent matchup."""
    started = time.monotonic()
    learned, model_kind = load_saved_agent(model_spec, device)
    opponent = make_agent(opponent_name, mc_rollouts)
    report = evaluate(
        learned, opponent, games=games, seed=seed,
        max_total_turns=max_turns, include_game_records=True,
    )
    records = report.pop("game_records")
    interval = paired_bootstrap_interval(
        records, samples=bootstrap_samples,
        seed=seed ^ sum(ord(char) for char in model_spec.name + opponent_name),
    )
    report["agents"][0]["name"] = model_spec.name
    report["agents"][1]["name"] = opponent_name
    report["agents"][0]["paired_bootstrap_win_rate95"] = interval
    report["model_file"] = str(model_spec.path)
    report["bootstrap_samples"] = bootstrap_samples
    key = f"{model_spec.name}_vs_{opponent_name}"
    write_csv(output_dir / f"{key}_games.csv", records)
    return {
        "key": key,
        "model": model_spec.name,
        "model_kind": model_kind,
        "opponent": opponent_name,
        "report": report,
        "interval": interval,
        "scores": scores_by_seed(records),
        "elapsed_seconds": time.monotonic() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=parse_model_spec, nargs="+", required=True,
        metavar="[NAME=]PATH",
    )
    parser.add_argument(
        "--ppo-pairing", type=parse_ppo_pairing, action="append", default=[],
        metavar="NAME=POLICY",
        help=("Override a named PPO model's frozen dice pairing with ga or couto; "
              "repeat for multiple model aliases"),
    )
    parser.add_argument("--opponents", default="rule28-smart,mc")
    parser.add_argument("--games", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=40_000_641)
    parser.add_argument("--max-turns", type=int, default=2_000)
    parser.add_argument(
        "--mc-rollouts", type=int, default=64,
        help="Rollouts per MC decision (default: 64; matches head-to-head evaluation)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "xpu"),
                        default="cpu")
    parser.add_argument(
        "--workers", type=int, default=1,
        help=("Parallel model-opponent processes; 1 runs sequentially "
              "(default: 1)"),
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/classical_ml/evaluation"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    override_names = [name for name, _ in args.ppo_pairing]
    if len(set(override_names)) != len(override_names):
        raise SystemExit("each --ppo-pairing model name may be specified only once")
    overrides = dict(args.ppo_pairing)
    model_names = {spec.name for spec in args.models}
    unknown_overrides = sorted(overrides.keys() - model_names)
    if unknown_overrides:
        raise SystemExit(
            "PPO pairing overrides reference unknown models: "
            + ", ".join(unknown_overrides)
        )
    args.models = [
        ModelSpec(spec.name, spec.path, overrides.get(spec.name))
        for spec in args.models
    ]
    try:
        opponents = parse_opponents(args.opponents)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    if min(args.games, args.max_turns, args.mc_rollouts, args.bootstrap_samples) < 1:
        raise SystemExit("game, turn, rollout, and bootstrap counts must be positive")
    if args.workers < 1:
        raise SystemExit("workers must be >= 1")
    if args.games % 2:
        raise SystemExit("games must be even so every dice seed has both seat assignments")
    missing = [str(spec.path) for spec in args.models if not spec.path.is_file()]
    if missing:
        raise SystemExit(f"Model files not found: {', '.join(missing)}")
    if len({spec.name for spec in args.models}) != len(args.models):
        raise SystemExit("model names must be unique; use NAME=PATH to disambiguate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    paired_scores: dict[tuple[str, str], dict[int, float]] = {}
    print("model                    opponent          win rate   paired bootstrap 95%")
    print("------------------------ ---------------- ---------- --------------------")
    model_metadata: dict[str, dict[str, str]] = {}
    matchup_specs = [
        (model_spec, opponent_name)
        for model_spec in args.models
        for opponent_name in opponents
    ]
    matchup_count = len(matchup_specs)

    if args.workers > 1 and matchup_count > 1:
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
                    _parallel_matchup, model_spec, opponent_name,
                    games=args.games, seed=args.seed, max_turns=args.max_turns,
                    mc_rollouts=args.mc_rollouts,
                    bootstrap_samples=args.bootstrap_samples,
                    device=args.device, output_dir=args.output_dir,
                ): (model_spec.name, opponent_name)
                for model_spec, opponent_name in matchup_specs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                model_key, opponent_name = futures[future]
                result = future.result()
                results_by_key[result["key"]] = result
                print(
                    f"Completed {completed}/{matchup_count}: "
                    f"{model_key} vs {opponent_name} "
                    f"({result['elapsed_seconds'] / 60:.1f} min)",
                    flush=True,
                )
        results = [
            results_by_key[f"{model_spec.name}_vs_{opponent_name}"]
            for model_spec, opponent_name in matchup_specs
        ]
    else:
        results = []
        for model_spec in args.models:
            learned, model_kind = load_saved_agent(model_spec, args.device)
            for opponent_name in opponents:
                started = time.monotonic()
                opponent = make_agent(opponent_name, args.mc_rollouts)
                report = evaluate(
                    learned, opponent, games=args.games, seed=args.seed,
                    max_total_turns=args.max_turns, include_game_records=True,
                )
                records = report.pop("game_records")
                interval = paired_bootstrap_interval(
                    records, samples=args.bootstrap_samples,
                    seed=(args.seed ^ sum(
                        ord(char) for char in model_spec.name + opponent_name
                    )),
                )
                report["agents"][0]["name"] = model_spec.name
                report["agents"][1]["name"] = opponent_name
                report["agents"][0]["paired_bootstrap_win_rate95"] = interval
                report["model_file"] = str(model_spec.path)
                report["bootstrap_samples"] = args.bootstrap_samples
                key = f"{model_spec.name}_vs_{opponent_name}"
                write_csv(args.output_dir / f"{key}_games.csv", records)
                results.append({
                    "key": key,
                    "model": model_spec.name,
                    "model_kind": model_kind,
                    "opponent": opponent_name,
                    "report": report,
                    "interval": interval,
                    "scores": scores_by_seed(records),
                    "elapsed_seconds": time.monotonic() - started,
                })

    for result in results:
        model_key = result["model"]
        opponent_name = result["opponent"]
        report = result["report"]
        interval = result["interval"]
        model_metadata[model_key] = {
            "kind": result["model_kind"],
            "path": str(next(
                spec.path for spec in args.models if spec.name == model_key
            )),
            "pairing_policy": next(
                spec.ppo_pairing_policy or "checkpoint-configured"
                for spec in args.models if spec.name == model_key
            ),
        }
        summaries[result["key"]] = report
        paired_scores[(model_key, opponent_name)] = result["scores"]
        rate = report["agents"][0]["win_rate"]
        print(
            f"{model_key:<24} {opponent_name:<16} {rate:>9.1%} "
            f"{interval[0]:>8.1%}-{interval[1]:<8.1%}"
        )

    comparisons: dict[str, dict] = {}
    model_keys = [spec.name for spec in args.models]
    for opponent_name in opponents:
        for first, second in itertools.combinations(model_keys, 2):
            estimate, low, high = paired_difference_interval(
                paired_scores[(first, opponent_name)],
                paired_scores[(second, opponent_name)],
                samples=args.bootstrap_samples,
                seed=args.seed ^ sum(ord(char) for char in first + second + opponent_name),
            )
            comparisons[f"{first}_minus_{second}_vs_{opponent_name}"] = {
                "first_model": first,
                "second_model": second,
                "opponent": opponent_name,
                "win_rate_difference": estimate,
                "paired_bootstrap_difference95": [low, high],
                "interval_excludes_zero": low > 0.0 or high < 0.0,
            }

    summary_path = args.output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps({
        "games_per_matchup": args.games,
        "seed": args.seed,
        "workers": min(args.workers, matchup_count),
        "pairing_policy": (
            "per-model; see models metadata" if overrides
            else "checkpoint-configured (fixed or learned by model)"
        ),
        "paired_bootstrap_unit": "shared_dice_seed_two_seat_games",
        "models": model_metadata,
        "matchups": summaries,
        "paired_model_comparisons": comparisons,
    }, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
