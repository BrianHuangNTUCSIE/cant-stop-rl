"""Mutable two-player Can't Stop state machine; contains no agent policy."""
from __future__ import annotations

import random
from typing import Sequence

from .constants import LENGTH
from .model import Move, State
from .rules import advance_active, legal_moves


class Game:
    """Explicit state machine: 'roll' -> 'pair' -> 'decision' -> 'roll'/next turn.

    The first roll of every turn is compulsory. After a legal move, the agent
    can bank the turn or request another roll. No artificial roll limit.
    """
    def __init__(self) -> None:
        self.permanent = [[0]*11 for _ in range(2)]
        self.owner = [-1]*11
        self.active: dict[int, int] = {}
        self.current_player = 0
        self.rolls_this_turn = 0
        self.phase = "roll"
        self.winner: int | None = None
        self.last_dice: tuple[int, ...] | None = None
        self.available: tuple[Move, ...] = ()
        self.turns = [0, 0]
        self.rolls = [0, 0]
        self.busts = [0, 0]
        self.stops = [0, 0]

    def snapshot(self) -> State:
        return State((tuple(self.permanent[0]), tuple(self.permanent[1])),
                     tuple(self.owner), tuple(sorted(self.active.items())),
                     self.current_player, self.rolls_this_turn)

    def _end_turn(self) -> None:
        self.active.clear()
        self.rolls_this_turn = 0
        self.current_player = 1 - self.current_player
        self.phase = "roll"
        self.last_dice = None
        self.available = ()

    def roll(self, rng: random.Random, dice: Sequence[int] | None = None) -> tuple[Move, ...]:
        if self.phase != "roll" or self.winner is not None:
            raise RuntimeError("roll() only allowed in the roll phase")
        p = self.current_player
        if self.rolls_this_turn == 0:
            self.turns[p] += 1
        dice = tuple(dice) if dice is not None else tuple(rng.randint(1, 6) for _ in range(4))
        choices = legal_moves(dice, self.permanent[p], self.owner, self.active)
        self.rolls[p] += 1
        self.last_dice = tuple(dice)
        if not choices:
            self.busts[p] += 1
            self._end_turn()
            return ()
        self.available = choices
        self.phase = "pair"
        return choices

    def apply_move(self, move: Move) -> None:
        if self.phase != "pair" or move not in self.available:
            raise ValueError("Expected a legal move after rolling")
        self.active = advance_active(self.active, self.permanent[self.current_player], move)
        self.rolls_this_turn += 1
        self.phase = "decision"
        self.available = ()

    def continue_turn(self) -> None:
        if self.phase != "decision":
            raise RuntimeError("continue_turn() only allowed after choosing a move")
        self.phase = "roll"

    def stop(self) -> None:
        if self.phase != "decision":
            raise RuntimeError("stop() only allowed after choosing a move")
        p = self.current_player
        for c, pos in self.active.items():
            self.permanent[p][c-2] = pos
            if pos == LENGTH[c]:
                self.owner[c-2] = p
        self.stops[p] += 1
        if self.owner.count(p) >= 3:
            self.winner = p
            self.phase = "done"
            self.active.clear()
            self.available = ()
        else:
            self._end_turn()
