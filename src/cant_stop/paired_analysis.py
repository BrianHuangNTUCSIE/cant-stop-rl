"""Dependency-free paired analysis for evaluator game records."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def _outcome(record: Mapping[str, object]) -> int:
    winner = record.get("winner")
    if winner == "A":
        return 1
    if winner == "B":
        return -1
    if winner == "draw":
        return 0
    raise ValueError(f"invalid game-record winner: {winner!r}")


def paired_units(records: Iterable[Mapping[str, object]]) -> list[tuple[int, ...]]:
    """Group signed game outcomes by dice seed.

    The evaluator's alternating-seat games with the same seed form one paired
    unit.  A final odd game is retained as a one-observation unit, which makes
    the function safe for an odd requested game count.
    """
    grouped: dict[object, list[int]] = defaultdict(list)
    for record in records:
        if "seed" not in record:
            raise ValueError("game record is missing seed")
        grouped[record["seed"]].append(_outcome(record))
    if not grouped:
        raise ValueError("at least one game record is required")
    return [tuple(outcomes) for _, outcomes in sorted(grouped.items(), key=lambda item: str(item[0]))]


def _margin(units: Sequence[Sequence[int]]) -> float:
    values = [value for unit in units for value in unit]
    return sum(values) / len(values)


def paired_bootstrap(
    records: Iterable[Mapping[str, object]],
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Estimate A-minus-B win-rate margin with paired resampling.

    Bootstrap units are dice-seed groups, not individual games.  The reported
    percentile interval is descriptive and should be interpreted alongside
    independent evaluation seeds when available.
    """
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    units = paired_units(records)
    observed = _margin(units)
    values = [value for unit in units for value in unit]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        sampled = [units[rng.randrange(len(units))] for _ in units]
        samples.append(_margin(sampled))
    samples.sort()
    lower = samples[int(0.025 * (resamples - 1))]
    upper = samples[int(0.975 * (resamples - 1))]

    # Under a no-difference null, swap A/B labels independently per paired
    # unit.  This preserves the paired magnitude and draw structure.
    extreme = 0
    for _ in range(resamples):
        null_total = 0
        for unit in units:
            sign = -1 if rng.getrandbits(1) else 1
            null_total += sign * sum(unit)
        if abs(null_total / sum(len(unit) for unit in units)) >= abs(observed):
            extreme += 1
    p_value = (extreme + 1) / (resamples + 1)
    return {
        "games": len(values),
        "paired_units": len(units),
        "a_wins": values.count(1),
        "b_wins": values.count(-1),
        "draws": values.count(0),
        "observed_margin_a_minus_b": observed,
        "bootstrap_percentile95": [lower, upper],
        "sign_flip_p_value": p_value,
        "resamples": resamples,
        "analysis_seed": seed,
        "unit_definition": "all records sharing evaluator dice seed",
    }
