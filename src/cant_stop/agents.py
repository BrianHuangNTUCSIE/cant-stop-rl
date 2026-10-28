"""Baseline pairing and Stop/Continue policies."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .constants import (
    CLASSIC_MOVE_WEIGHT,
    COLUMNS,
    GA_MOVE_WEIGHT,
    LENGTH,
    MARK_VALUE,
    STEP_VALUE,
)
from .model import Move, State
from .rules import (
    _legal_move_columns_fast,
    advance_active,
    pair_score,
    reached_top,
    rule28_points,
)


class Agent(Protocol):
    def choose_move(self, state: State, moves: Sequence[Move], rng: random.Random) -> Move: ...
    def should_stop(self, state: State, rng: random.Random) -> bool: ...


def _best_pair_moves(
    state: State, moves: Sequence[Move], heuristic: str,
) -> list[Move]:
    """Return maximum-scoring moves while computing board data once.

    This is equivalent to ``max(pair_score(...))`` followed by a second
    scoring pass, including the original move-order tie behavior.  It is used
    in rollout-heavy policies where repeatedly rebuilding ``active`` and
    rescoring every move is a measurable cost.
    """
    if heuristic == "ga":
        weights = GA_MOVE_WEIGHT
    elif heuristic == "classic":
        weights = CLASSIC_MOVE_WEIGHT
    else:
        raise ValueError(f"Unknown pair heuristic {heuristic!r}")
    active = state.active_dict()
    top = float("-inf")
    best: list[Move] = []
    for move in moves:
        score = (
            sum(weights[column] for column in move.columns)
            - 6 * len({column for column in move.columns if column not in active})
        )
        if score > top:
            top = score
            best = [move]
        elif score == top:
            best.append(move)
    return best


@dataclass
class RandomAgent:
    stop_probability: float = 0.3

    def choose_move(self, state: State, moves: Sequence[Move], rng: random.Random) -> Move:
        return rng.choice(moves)

    def should_stop(self, state: State, rng: random.Random) -> bool:
        return rng.random() < self.stop_probability


@dataclass
class Rule28Agent:
    pair_heuristic: str = "ga"
    smart: bool = True
    threshold: int = 28

    def choose_move(self, state: State, moves: Sequence[Move], rng: random.Random) -> Move:
        # Randomized tie breaking prevents systematically favoring lower columns.
        return rng.choice(_best_pair_moves(state, moves, self.pair_heuristic))

    def should_stop(self, state: State, rng: random.Random) -> bool:
        if self.smart and reached_top(state):
            return True
        if len(state.active) < 3:
            return False  # historical pure Rule of 28 does not stop early
        return rule28_points(state) >= self.threshold


# Parameters used by Glenn & Aloi's optimized Rule-of-28 program and by the
# Sketch-SA(O) program synthesized from it.  Keeping these here, rather than in
# the game rules, makes clear that they are policy choices.
_PUBLISHED_PROGRESS_WEIGHT = dict(zip(COLUMNS, (7, 7, 3, 2, 2, 1, 2, 2, 3, 7, 7)))


def _wins_after_stop(state: State) -> bool:
    temporary_columns = sum(pos == LENGTH[column] for column, pos in state.active)
    return state.completed(state.current_player) + temporary_columns >= 3


def _neutral_columns(state: State) -> tuple[int, ...]:
    """Columns represented as neutral markers in the reference simulator.

    The engine retains a top-reaching neutral until the player banks. The
    published policies also observe that neutral during the remainder of the
    turn, so it remains in this translated view.
    """
    return tuple(column for column, _ in state.active)


def _difficulty_score(columns: Sequence[int]) -> int:
    # Deliberately retain Python's all([]) behavior used by the reference code.
    score = 0
    if all(column % 2 != 0 for column in columns):
        score += 7
    if all(column % 2 == 0 for column in columns):
        score += 1
    if all(column <= 7 for column in columns):
        score += 5
    if all(column >= 7 for column in columns):
        score += 6
    return score


def _advanced_this_turn(state: State, column: int) -> int:
    """Port the reference policy's board-index calculation.

    Its simulator moves a top-reaching neutral into a temporary-won list and
    awards one extra virtual cell when the player had banked progress there.
    """
    active = state.active_dict()
    position = active.get(column)
    if position is None:
        return 0
    banked = state.permanent[state.current_player][column - 2]
    return position - banked


def _secured_position(state: State, column: int) -> int:
    """Return Couto's f3 feature in the reference simulator's coordinates."""
    if state.owner[column - 2] != -1:
        return LENGTH[column] + 1 if state.owner[column - 2] == state.current_player else 0
    position = state.active_dict().get(column)
    if position is None:
        return state.permanent[state.current_player][column - 2]
    return position


@dataclass
class GlennAloiAgent:
    """Full Glenn & Aloi optimized heuristic as released by Medeiros et al.

    This includes optimized progress weights, difficulty bonuses, threshold 29,
    and the published constant-weight pairing policy.  It is distinct from the
    project's simpler ``rule28-smart`` hybrid.
    """

    threshold: int = 29

    def choose_move(self, state: State, moves: Sequence[Move], rng: random.Random) -> Move:
        active = state.active_dict()

        def score(move: Move) -> int:
            advance = 2 if len(move.columns) == 2 and move.columns[0] == move.columns[1] else 1
            return sum(
                advance * GA_MOVE_WEIGHT[column] - 6 * int(column not in active)
                for column in move.columns
            )

        # np.argmax in the released implementation chooses the first maximum.
        return max(moves, key=score)

    def should_stop(self, state: State, rng: random.Random) -> bool:
        if _wins_after_stop(state):
            return True
        # The reference policy always continues while an unused neutral marker
        # and at least one unfinished column remain.
        if len(state.active) < 3 and any(owner == -1 for owner in state.owner):
            return False
        neutrals = _neutral_columns(state)
        score = sum(
            (_advanced_this_turn(state, column) + 1) * _PUBLISHED_PROGRESS_WEIGHT[column]
            for column in neutrals
        )
        return score + _difficulty_score(neutrals) >= self.threshold


def _paper_difficulty(columns: Sequence[int], *, odds: int, evens: int,
                      highs: int, lows: int) -> int:
    """Difficulty bonuses from the Glenn heuristic papers.

    The conditions are intentionally independent: column 7 satisfies both the
    high and low conditions, as it does in the published scoring rule.
    """
    if not columns:
        return 0
    score = 0
    if all(column % 2 for column in columns):
        score += odds
    if all(column % 2 == 0 for column in columns):
        score += evens
    if all(column >= 7 for column in columns):
        score += highs
    if all(column <= 7 for column in columns):
        score += lows
    return score


class _GlennPaperAgent:
    """Shared implementation for the paper-backed Glenn heuristics."""

    progress_weights: dict[int, int]
    threshold: int
    odds: int
    evens: int
    highs: int
    lows: int
    marker_penalty: int

    def _stop_score(self, state: State) -> int:
        active = state.active_dict()
        player_progress = state.permanent[state.current_player]
        columns = tuple(active)
        progress = sum(
            (active[column] - player_progress[column - 2] + 1)
            * self.progress_weights[column]
            for column in columns
        )
        return progress + _paper_difficulty(
            columns, odds=self.odds, evens=self.evens,
            highs=self.highs, lows=self.lows,
        )

    def _move_score(self, state: State, move: Move) -> int:
        active = state.active_dict()
        permanent = state.permanent[state.current_player]
        next_positions = active.copy()
        for column in move.columns:
            next_positions[column] = next_positions.get(
                column, permanent[column - 2]
            ) + 1
        return self._move_value(state, move, next_positions) - self.marker_penalty * len(
            {column for column in move.columns if column not in active}
        )

    def choose_move(self, state: State, moves: Sequence[Move],
                    rng: random.Random) -> Move:
        # The papers do not specify randomized tie-breaking. Python's max keeps
        # the engine's deterministic legal-move order on ties.
        return max(moves, key=lambda move: self._move_score(state, move))

    def should_stop(self, state: State, rng: random.Random) -> bool:
        if _wins_after_stop(state):
            return True
        return self._stop_score(state) >= self.threshold

    def _move_value(self, state: State, move: Move,
                    next_positions: dict[int, int]) -> int:
        raise NotImplementedError


@dataclass
class Glenn2009ConstantAgent(_GlennPaperAgent):
    """Clean-room implementation of Glenn & Aloi's 2009 Constant champion.

    This is classified as ``paper_reimplementation``. It uses the current
    engine's legal moves and banking semantics; any simulator convention that
    differs from the paper is therefore an explicit engine adaptation.
    """

    progress_weights: dict[int, int] = field(default_factory=lambda: dict(
        zip(COLUMNS, (7, 7, 3, 2, 2, 1, 2, 2, 3, 7, 7))
    ))
    move_weights: dict[int, int] = field(default_factory=lambda: dict(
        zip(COLUMNS, (7, 0, 2, 0, 4, 3, 4, 0, 2, 0, 7))
    ))
    threshold: int = 29
    odds: int = 7
    evens: int = 1
    highs: int = 6
    lows: int = 5
    marker_penalty: int = 6

    def _move_value(self, state: State, move: Move,
                    next_positions: dict[int, int]) -> int:
        return sum(
            self.move_weights[column]
            for column in move.columns
        )


@dataclass
class Glenn2010LinearAgent(_GlennPaperAgent):
    """Clean-room implementation of the verified 2010 Linear champion.

    The paper defines each move-space weight as
    ``floor(m[column] * j / column_height + b[column])``. This uses the
    Table 6 champion, not the conflicting transcription formerly present in
    the implementation guide.
    """

    progress_weights: dict[int, int] = field(default_factory=lambda: dict(
        zip(COLUMNS, (15, 13, 8, 8, 4, 3, 4, 8, 8, 13, 15))
    ))
    slopes: dict[int, int] = field(default_factory=lambda: dict(
        zip(COLUMNS, (72, 48, 56, 44, 28, 32, 28, 44, 56, 48, 72))
    ))
    intercepts: dict[int, int] = field(default_factory=lambda: dict(
        zip(COLUMNS, (15, 0, 3, 0, 7, 5, 7, 0, 3, 0, 15))
    ))
    threshold: int = 61
    odds: int = 11
    evens: int = -1
    highs: int = 4
    lows: int = 4
    marker_penalty: int = 12

    def _move_value(self, state: State, move: Move,
                    next_positions: dict[int, int]) -> int:
        active = state.active_dict()
        return sum(
            int(self.slopes[column] * position / LENGTH[column]
                + self.intercepts[column])
            for column in set(move.columns)
            for position in range(
                active.get(column, state.permanent[state.current_player][column - 2]) + 1,
                next_positions[column] + 1,
            )
        )


@dataclass
class CoutoAgent:
    """Sketch-SA(O), the strongest released Can't Stop policy from Couto et al.

    ``pair_heuristic='couto'`` is the original full policy.  ``'ga'`` keeps the
    project's fixed Glenn–Aloi pairing so experiments isolate Stop/Continue;
    that mode is a project adaptation, not the paper's original agent.
    """

    pair_heuristic: str = "couto"
    threshold: int = 29

    def choose_move(self, state: State, moves: Sequence[Move], rng: random.Random) -> Move:
        if self.pair_heuristic == "ga":
            return rng.choice(_best_pair_moves(state, moves, "ga"))
        if self.pair_heuristic != "couto":
            raise ValueError("pair_heuristic must be 'couto' or 'ga'")

        def score(move: Move) -> int:
            return sum(
                _secured_position(state, column) + GA_MOVE_WEIGHT[column]
                for column in move.columns
            )

        return max(moves, key=score)

    def should_stop(self, state: State, rng: random.Random) -> bool:
        if _wins_after_stop(state):
            return True
        if len(state.active) < 3 and any(owner == -1 for owner in state.owner):
            return False
        neutrals = _neutral_columns(state)
        score = -len(neutrals) * sum(neutrals)
        score += sum(
            _PUBLISHED_PROGRESS_WEIGHT[column]
            * _advanced_this_turn(state, column)
            * _secured_position(state, column)
            for column in neutrals
        )
        difficulty = _difficulty_score(neutrals)
        return score + difficulty * difficulty + difficulty >= self.threshold


@dataclass
class MonteCarloAgent:
    """One-turn Monte Carlo rollouts for stop/continue, NOT full-game MCTS.

    Assumes a fixed GA pairing heuristic and Rule-of-28 continuation policy.
    Compares bank-now value to average bank-or-bust value after forcing one
    additional roll. Simulator uses identical game move legality as Game.
    """
    rollouts: int = 64
    future_roll_limit: int = 40
    pair_heuristic: str = "ga"

    def choose_move(self, state: State, moves: Sequence[Move], rng: random.Random) -> Move:
        return rng.choice(_best_pair_moves(state, moves, self.pair_heuristic))

    @staticmethod
    def _bank_value(state: State, active: dict[int, int]) -> float:
        p = state.current_player
        perm = state.permanent[p]
        # Incremental banked progress + completion bonus, relative to bust (=0).
        value = sum((pos - perm[c-2]) / LENGTH[c] for c, pos in active.items())
        claimed = sum(pos == LENGTH[c] for c, pos in active.items())
        value += claimed * 1.5
        if state.completed(p) + claimed >= 3:
            value += 20.0
        return value

    def _rollout(self, initial: State, rng: random.Random) -> float:
        perm = initial.permanent[initial.current_player]
        active = initial.active_dict()
        # The fallback policy only observes these three pieces of state.  Keep
        # the rollout in its compact representation instead of constructing
        # two immutable State snapshots for every simulated roll.
        pair_heuristic = self.pair_heuristic
        if pair_heuristic == "ga":
            weights = GA_MOVE_WEIGHT
        elif pair_heuristic == "classic":
            weights = CLASSIC_MOVE_WEIGHT
        else:
            raise ValueError(f"Unknown pair heuristic {pair_heuristic!r}")
        randint = rng.randint
        for _ in range(self.future_roll_limit):
            # Keep the same four randint calls and order, while avoiding a
            # bound-method lookup for every die in the rollout hot path.
            dice = (randint(1, 6), randint(1, 6), randint(1, 6), randint(1, 6))
            moves = _legal_move_columns_fast(dice, perm, initial.owner, active)
            if not moves:
                return 0.0  # bust: the current turn contributes no banked progress
            # This is Rule28Agent.choose_move for the fixed pairing.  Preserve
            # sorted candidate order so seeded tie-breaking remains identical.
            top = float("-inf")
            best: list[tuple[int, ...]] = []
            for columns in moves:
                score = (
                    sum(weights[column] for column in columns)
                    - 6 * len({column for column in columns if column not in active})
                )
                if score > top:
                    top = score
                    best = [columns]
                elif score == top:
                    best.append(columns)
            selected = rng.choice(best)
            active = active.copy()
            for column in selected:
                active[column] = active.get(column, perm[column - 2]) + 1
            # This is Rule28Agent.should_stop(smart=True), expressed directly
            # on the same state fields.  It does not consume RNG.
            if any(pos == LENGTH[column] for column, pos in active.items()):
                should_stop = True
            elif len(active) < 3:
                should_stop = False
            else:
                points = 0
                for column, pos in active.items():
                    steps = pos - perm[column - 2]
                    points += MARK_VALUE[column] + (steps - 1) * STEP_VALUE[column]
                columns = tuple(active)
                if all(column % 2 == 1 for column in columns):
                    points += 2
                if all(column % 2 == 0 for column in columns):
                    points -= 2
                if all(column < 8 for column in columns):
                    points += 4
                if all(column > 6 for column in columns):
                    points += 4
                should_stop = points >= 28
            if should_stop:
                return self._bank_value(initial, active)
        # Explicit truncation: bank at the horizon rather than pretending to win.
        return self._bank_value(initial, active)

    def should_stop(self, state: State, rng: random.Random) -> bool:
        if self.rollouts < 1 or self.future_roll_limit < 1:
            raise ValueError("rollouts and future_roll_limit must be positive")
        if state.completed(state.current_player) + sum(pos == LENGTH[c] for c, pos in state.active) >= 3:
            return True
        bank_now = self._bank_value(state, state.active_dict())
        continue_value = sum(self._rollout(state, rng) for _ in range(self.rollouts)) / self.rollouts
        return bank_now >= continue_value


@dataclass
class StrongMonteCarloAgent(MonteCarloAgent):
    """Fast opponent-aware one-turn Monte Carlo policy.

    This keeps the original MC computation budget and GA pairing, but scores
    candidate banking outcomes using the game race as well as local progress.
    It is intentionally not a full-game rollout: the previous
    ``mc-full-game`` experiment was both much slower and weaker in the smoke
    comparison.  The original ``mc`` remains unchanged for reproducibility.
    """

    race_weight: float = 1.0

    def _bank_value(self, state: State, active: dict[int, int]) -> float:
        value = super()._bank_value(state, active)
        player = state.current_player
        opponent = 1 - player
        claimed = sum(pos == LENGTH[column] for column, pos in active.items())
        completed_after_bank = state.completed(player) + claimed
        opponent_completed = state.completed(opponent)
        own_progress = sum(
            (pos - state.permanent[player][column - 2]) / LENGTH[column]
            for column, pos in active.items()
        )
        opponent_progress = sum(
            state.permanent[opponent][index] / LENGTH[column]
            for index, column in enumerate(range(2, 13))
        ) / 11.0
        # A bank is more valuable when it preserves a lead and when the
        # opponent is close to winning.  The same score is used for bank-now
        # and rollout outcomes, so it changes the stopping boundary rather
        # than merely adding a constant to one side.
        race = self.race_weight * (
            1.5 * (completed_after_bank - opponent_completed)
            + 0.5 * (own_progress - opponent_progress)
        )
        if opponent_completed >= 2 and completed_after_bank < 2:
            race -= self.race_weight
        return value + race


@dataclass
class FullGameMonteCarloAgent:
    """Full-game Monte Carlo Stop/Continue policy.

    Unlike :class:`MonteCarloAgent`, each sample continues through the rest of
    the game and returns a terminal win value.  This is deliberately a
    separate agent: the original ``mc`` remains the project's one-turn
    approximate baseline.  Pairing and rollout continuation use the supplied
    heuristic, so this is still a rollout policy rather than full MCTS.
    """

    rollouts: int = 256
    pair_heuristic: str = "ga"
    max_turns: int = 2000

    def choose_move(self, state: State, moves: Sequence[Move], rng: random.Random) -> Move:
        return rng.choice(_best_pair_moves(state, moves, self.pair_heuristic))

    @staticmethod
    def _copy_game(state: State) -> "Game":
        # Local import avoids the agents -> game -> rules import cycle.
        from .game import Game

        game = Game()
        game.permanent = [list(state.permanent[0]), list(state.permanent[1])]
        game.owner = list(state.owner)
        game.active = state.active_dict()
        game.current_player = state.current_player
        game.rolls_this_turn = state.rolls_this_turn
        game.columns_started = set(state.columns_started)
        game.phase = "decision"
        return game

    def _rollout(self, state: State, first_action: int,
                 rng: random.Random) -> float:
        """Return terminal win value for the player at the root state."""
        from .game import Game

        game = self._copy_game(state)
        root_player = state.current_player
        if first_action:
            game.stop()
        else:
            game.continue_turn()

        policy = Rule28Agent(pair_heuristic=self.pair_heuristic, smart=True)
        turns = 0
        while game.winner is None and turns < self.max_turns:
            if game.phase == "roll":
                moves = game.roll(rng)
                if not moves:
                    continue
            if game.phase == "pair":
                selected = policy.choose_move(
                    game.snapshot(), game.available, rng
                )
                game.apply_move(selected)
            if game.phase == "decision":
                if policy.should_stop(game.snapshot(), rng):
                    game.stop()
                else:
                    game.continue_turn()
            turns = sum(game.turns)
        if game.winner is None:
            return 0.5
        return 1.0 if game.winner == root_player else 0.0

    def should_stop(self, state: State, rng: random.Random) -> bool:
        if self.rollouts < 1 or self.max_turns < 1:
            raise ValueError("rollouts and max_turns must be positive")
        # A winning bank is always optimal and avoids unnecessary simulation.
        if _wins_after_stop(state):
            return True
        stop_value = sum(
            self._rollout(state, 1, rng) for _ in range(self.rollouts)
        ) / self.rollouts
        continue_value = sum(
            self._rollout(state, 0, rng) for _ in range(self.rollouts)
        ) / self.rollouts
        return stop_value >= continue_value


def _copy_decision_game(state: State):
    """Create a private Game positioned immediately after a move.

    Search agents must never mutate the live evaluator game.  The import is
    local because Game depends on the Agent-facing State type.
    """
    from .game import Game

    game = Game()
    game.permanent = [list(state.permanent[0]), list(state.permanent[1])]
    game.owner = list(state.owner)
    game.active = state.active_dict()
    game.current_player = state.current_player
    game.rolls_this_turn = state.rolls_this_turn
    game.columns_started = set(state.columns_started)
    game.phase = "decision"
    return game


def _random_full_game_rollout(state: State, action: int, rng: random.Random,
                              max_total_turns: int) -> int:
    """Run one private random rollout and return its winner.

    ``action`` follows vmerckle's convention: 0 banks and 1 continues.  The
    original AIrand chooses both a legal pairing and this binary action
    uniformly at every decision, so this is deliberately not Rule-of-28.
    """
    game = _copy_decision_game(state)
    if action == 0:
        game.stop()
    else:
        game.continue_turn()
    while game.winner is None:
        if game.phase == "roll" and game.rolls_this_turn == 0 \
                and sum(game.turns) >= max_total_turns:
            break
        if game.phase == "roll":
            if not game.roll(rng):
                continue
        if game.phase == "pair":
            game.apply_move(rng.choice(game.available))
        if game.phase == "decision":
            if rng.randrange(2) == 0:
                game.stop()
            else:
                game.continue_turn()
    return game.winner if game.winner is not None else -1


@dataclass
class _VMerckleSearchAgent:
    """Clean-room root search adapter based on vmerckle/CantStop-AI.

    The reference source has no license and is therefore not copied.  Its
    `AIrand` rollout semantics and action-selection equations are reproduced
    independently against this project's rules.  This is not full MCTS: the
    search has a single root and does not retain a tree of descendants.
    """

    rollouts: int = 64
    max_total_turns: int = 2000
    _pending_stop: dict[State, bool] = field(default_factory=dict, init=False,
                                             repr=False)

    def _actions(self, state: State, moves: Sequence[Move]) -> list[tuple[Move, int]]:
        return [(move, action) for move in moves for action in (0, 1)]

    def _post_move(self, state: State, move: Move) -> State:
        return State(
            state.permanent,
            state.owner,
            tuple(sorted(advance_active(state.active_dict(),
                                        state.permanent[state.current_player],
                                        move).items())),
            state.current_player,
            state.rolls_this_turn + 1,
            state.columns_started,
        )

    def _win_rate(self, state: State, action: tuple[Move, int],
                  rng: random.Random, count: int) -> int:
        post_move = self._post_move(state, action[0])
        wins = 0
        root_player = state.current_player
        for _ in range(count):
            if _random_full_game_rollout(post_move, action[1], rng,
                                         self.max_total_turns) == root_player:
                wins += 1
        return wins

    def _select(self, state: State, actions: list[tuple[Move, int]],
                rng: random.Random) -> tuple[Move, int]:
        raise NotImplementedError

    def choose_move(self, state: State, moves: Sequence[Move],
                    rng: random.Random) -> Move:
        if self.rollouts < 1 or self.max_total_turns < 1:
            raise ValueError("rollouts and max_total_turns must be positive")
        actions = self._actions(state, moves)
        selected = self._select(state, actions, rng)
        post_move = self._post_move(state, selected[0])
        self._pending_stop[post_move] = selected[1] == 0
        return selected[0]

    def should_stop(self, state: State, rng: random.Random) -> bool:
        # A winning bank is mandatory for a useful search policy and avoids a
        # pointless rollout whose continuation cannot improve the result.
        if _wins_after_stop(state):
            return True
        return self._pending_stop.pop(state, False)


@dataclass
class VMerckleMonteAgent(_VMerckleSearchAgent):
    """Flat root Monte Carlo policy (`vmerckle_monte`)."""

    def _select(self, state: State, actions: list[tuple[Move, int]],
                rng: random.Random) -> tuple[Move, int]:
        quota = self.rollouts // len(actions)
        best_index = 0
        best_score = 0
        if quota == 0:
            return actions[0]
        for index, action in enumerate(actions):
            score = self._win_rate(state, action, rng, quota)
            if index == 0 or score > best_score:
                best_index, best_score = index, score
        return actions[best_index]


@dataclass
class VMerckleMonteUCTAgent(_VMerckleSearchAgent):
    """Root-only UCB1 bandit policy (`vmerckle_monte_uct`)."""

    exploration: float = 1.0

    def _select(self, state: State, actions: list[tuple[Move, int]],
                rng: random.Random) -> tuple[Move, int]:
        if self.exploration < 0:
            raise ValueError("exploration must be non-negative")
        plays = [0] * len(actions)
        wins = [0] * len(actions)
        for quota in range(self.rollouts):
            index = next((i for i, count in enumerate(plays) if count == 0), None)
            if index is None:
                index = max(
                    range(len(actions)),
                    key=lambda i: wins[i] / plays[i]
                    + self.exploration * (math.log(quota) / plays[i]) ** 0.5,
                )
            wins[index] += self._win_rate(state, actions[index], rng, 1)
            plays[index] += 1
        played = [i for i, count in enumerate(plays) if count]
        best = max(played, key=lambda i: wins[i] / plays[i])
        return actions[best]


AGENT_NAMES = (
    "random", "rule28", "rule28-smart", "rule28-ga-pure", "mc",
    "mc-full-game", "mc-strong", "vmerckle_monte", "vmerckle_monte_uct",
    "glenn-aloi", "glenn2009_constant", "glenn2010_linear", "couto",
    "couto-ga-stop",
)


def make_agent(name: str, mc_rollouts: int = 64,
               mc_future_roll_limit: int = 40) -> Agent:
    choices: dict[str, Agent] = {
        "random": RandomAgent(),
        "rule28": Rule28Agent(pair_heuristic="classic", smart=False),
        "rule28-smart": Rule28Agent(pair_heuristic="ga", smart=True),
        "rule28-ga-pure": Rule28Agent(pair_heuristic="ga", smart=False),
        "mc": MonteCarloAgent(
            rollouts=mc_rollouts, future_roll_limit=mc_future_roll_limit,
        ),
        "mc-full-game": FullGameMonteCarloAgent(rollouts=mc_rollouts),
        "mc-strong": StrongMonteCarloAgent(
            rollouts=mc_rollouts, future_roll_limit=mc_future_roll_limit,
        ),
        "vmerckle_monte": VMerckleMonteAgent(rollouts=mc_rollouts),
        "vmerckle_monte_uct": VMerckleMonteUCTAgent(rollouts=mc_rollouts),
        "glenn-aloi": GlennAloiAgent(),
        "glenn2009_constant": Glenn2009ConstantAgent(),
        "glenn2010_linear": Glenn2010LinearAgent(),
        "couto": CoutoAgent(),
        "couto-ga-stop": CoutoAgent(pair_heuristic="ga"),
    }
    if name not in choices:
        raise ValueError(f"Unknown agent: {name}. Choose from {', '.join(choices)}")
    return choices[name]
