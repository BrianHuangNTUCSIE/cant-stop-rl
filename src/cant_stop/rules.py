"""Pure move legality, feature helpers, and heuristic scoring."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product
from typing import Sequence

from .constants import COLUMNS, LENGTH, GA_MOVE_WEIGHT, CLASSIC_MOVE_WEIGHT, MARK_VALUE, STEP_VALUE
from .model import Move, State


def _roll_sum_mask(dice: tuple[int, int, int, int]) -> int:
    """Bit mask of every column obtainable from any pairing of four dice."""
    a, b, c, d = dice
    sums = (a+b, c+d, a+c, b+d, a+d, b+c)
    return sum(1 << (column - 2) for column in set(sums))


# Many of the 1,296 ordered rolls have the same set of possible column sums.
# Compress them once so enhanced features do not repeatedly construct Moves or
# call legal_moves for every roll at every learner decision.
_ROLL_MASK_COUNTS = tuple(Counter(
    _roll_sum_mask(dice)
    for dice in product(range(1, 7), repeat=4)
).items())


def _can_step(column: int, permanent: Sequence[int], owner: Sequence[int],
              active: dict[int, int]) -> bool:
    if owner[column - 2] != -1:
        return False
    if column in active:
        return active[column] < LENGTH[column]
    return len(active) < 3 and permanent[column - 2] < LENGTH[column]


def _legal_move_columns(dice: Sequence[int], permanent: Sequence[int],
                        owner: Sequence[int], active: dict[int, int]) -> tuple[tuple[int, ...], ...]:
    """Return legal outcomes in dice-partition enumeration order.

    This is shared with the hot Monte Carlo rollout path.  Keeping the legality
    logic here avoids a faster-but-different simulator while letting rollouts
    operate on compact column tuples.

    Outcomes respect mandatory use of both sums.  The order is significant for
    published policies that use stable first-match tie breaking.

    When there is one free runner and two different new sums, both sums are
    individually usable but not together; expose each single-sum choice.
    """
    if len(dice) != 4 or any(not 1 <= x <= 6 for x in dice):
        raise ValueError("dice must contain exactly four integers in [1, 6]")
    return _legal_move_columns_fast(dice, permanent, owner, active)


def _legal_move_columns_fast(dice: Sequence[int], permanent: Sequence[int],
                             owner: Sequence[int], active: dict[int, int]) -> tuple[tuple[int, ...], ...]:
    """Unchecked legality path for trusted rollout dice.

    ``MonteCarloAgent`` generates validated dice itself.  Avoiding public API
    validation and using a tiny list for at most six outcomes removes hashing
    and temporary set allocations from the rollout hot path.  The returned
    order follows dice-partition enumeration and is duplicate-free.
    """
    a, b, c, d = dice
    partitions = ((a+b, c+d), (a+c, b+d), (a+d, b+c))
    active_count = len(active)
    choices: list[tuple[int, ...]] = []

    def can_step(column: int) -> bool:
        if owner[column - 2] != -1:
            return False
        position = active.get(column)
        if position is not None:
            return position < LENGTH[column]
        return active_count < 3 and permanent[column - 2] < LENGTH[column]

    def add_choice(choice: tuple[int, ...]) -> None:
        if choice not in choices:
            choices.append(choice)

    for x, y in partitions:
        if x == y:
            if not can_step(x):
                continue
            trial = active.copy()
            trial[x] = trial.get(x, permanent[x-2]) + 1
            # The trial has at most the same number of active columns, so the
            # only changed legality component is this column's position.
            if owner[x - 2] == -1 and trial[x] < LENGTH[x]:
                add_choice((x, x))
            else:
                add_choice((x,))
            continue

        can_x = can_step(x)
        can_y = can_step(y)
        if can_x and can_y:
            if active_count + int(x not in active) + int(y not in active) <= 3:
                add_choice(tuple(sorted((x, y))))
            else:
                add_choice((x,))
                add_choice((y,))
        elif can_x:
            add_choice((x,))
        elif can_y:
            add_choice((y,))
    return tuple(choices)


def legal_moves(dice: Sequence[int], permanent: Sequence[int],
                owner: Sequence[int], active: dict[int, int]) -> tuple[Move, ...]:
    """Enumerate distinct legal ``Move`` values for the public game API."""
    return tuple(
        Move(columns)
        for columns in _legal_move_columns(dice, permanent, owner, active)
    )


def advance_active(active: dict[int, int], permanent: Sequence[int],
                   move: Move) -> dict[int, int]:
    next_active = active.copy()
    for c in move.columns:
        next_active[c] = next_active.get(c, permanent[c-2]) + 1
    return next_active


@lru_cache(maxsize=1 << len(COLUMNS))
def _bust_probability_for_available_mask(available_mask: int) -> float:
    """Return exact bust probability for a set of stepable columns.

    A roll is non-busting exactly when at least one sum from one of its three
    pairings can advance. Whether a column can advance captures all state needed
    for that test, so the full board position does not belong in the cache key.
    """
    busts = sum(
        count for roll_mask, count in _ROLL_MASK_COUNTS
        if not roll_mask & available_mask
    )
    return busts / 1296.0


def bust_probability(state: State) -> float:
    permanent = state.permanent[state.current_player]
    active = state.active_dict()
    available_mask = 0
    for column in COLUMNS:
        if _can_step(column, permanent, state.owner, active):
            available_mask |= 1 << (column - 2)
    return _bust_probability_for_available_mask(available_mask)


def rule28_points(state: State) -> int:
    """Keller's Rule of 28: initial mark scores 2x later advancement."""
    active = state.active_dict()
    me = state.permanent[state.current_player]
    value = 0
    for c, pos in active.items():
        steps = pos - me[c-2]
        assert steps >= 1
        value += MARK_VALUE[c] + (steps - 1) * STEP_VALUE[c]
    if len(active) == 3:
        cols = tuple(active)
        if all(c % 2 == 1 for c in cols):
            value += 2
        if all(c % 2 == 0 for c in cols):
            value -= 2
        if all(c < 8 for c in cols):
            value += 4
        if all(c > 6 for c in cols):
            value += 4
    return value


def reached_top(state: State) -> bool:
    return any(pos == LENGTH[c] for c, pos in state.active)


def pair_score(state: State, move: Move, kind: str = "ga") -> float:
    """Glenn & Aloi constant-weight move score or simple center preference."""
    weights = GA_MOVE_WEIGHT if kind == "ga" else CLASSIC_MOVE_WEIGHT
    if kind not in ("ga", "classic"):
        raise ValueError(f"Unknown pair heuristic {kind!r}")
    active = state.active_dict()
    new_markers = sum(c not in active for c in set(move.columns))
    return sum(weights[c] for c in move.columns) - 6 * new_markers
