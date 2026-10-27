"""Seeded game execution and head-to-head evaluation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from .agents import Agent
from .game import Game


@dataclass(frozen=True)
class GameResult:
    winner: int | None
    turns: tuple[int, int]
    rolls: tuple[int, int]
    busts: tuple[int, int]
    stops: tuple[int, int]


def play_game(agents: Sequence[Agent], seed: int, max_total_turns: int = 2000) -> GameResult:
    if len(agents) != 2:
        raise ValueError("Exactly two agents are required")
    game = Game()
    dice_rng = random.Random(seed)
    agent_rng = (random.Random(seed ^ 0x1234ABCD), random.Random(seed ^ 0xFEDC5678))
    if max_total_turns < 1:
        raise ValueError("max_total_turns must be >= 1")
    while game.winner is None:
        # Finish the last permitted turn; never truncate it mid-roll.
        if (game.phase == "roll" and game.rolls_this_turn == 0
                and sum(game.turns) >= max_total_turns):
            break
        player = game.current_player
        agent = agents[player]
        if game.phase == "roll":
            moves = game.roll(dice_rng)
            if not moves:
                continue
        if game.phase == "pair":
            state = game.snapshot()
            move = agent.choose_move(state, game.available, agent_rng[player])
            game.apply_move(move)
        if game.phase == "decision":
            state = game.snapshot()
            if agent.should_stop(state, agent_rng[player]):
                game.stop()
            else:
                game.continue_turn()
    # If max_total_turns was reached in the middle of a turn, return a draw.
    return GameResult(game.winner, tuple(game.turns), tuple(game.rolls),
                      tuple(game.busts), tuple(game.stops))


def wilson_interval(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for Bernoulli win/non-win outcomes."""
    if n < 1:
        return (0.0, 1.0)
    phat = wins/n
    denominator = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denominator
    radius = z/denominator * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))
    return (max(0.0, center-radius), min(1.0, center+radius))


def evaluate(agent_a: Agent, agent_b: Agent, games: int = 1000,
             seed: int = 42, max_total_turns: int = 2000,
             include_game_records: bool = False,
             progress_callback: Callable[[int, int], None] | None = None) -> dict:
    """Alternate seats each game, same seed for both seat assignments in a pair.

    CI treats games as independent (only approximate for paired seeds). For
    rigorous inference compare independent multi-seed batches or bootstrap pairs.
    """
    if games < 1:
        raise ValueError("games must be >= 1")
    wins = [0, 0]
    turns = [0, 0]
    rolls = [0, 0]
    busts = [0, 0]
    stops = [0, 0]
    seat_wins = [[0, 0], [0, 0]]
    draws = 0
    records: list[dict] = []
    for i in range(games):
        swapped = i % 2 == 1
        lineup = (agent_b, agent_a) if swapped else (agent_a, agent_b)
        result = play_game(lineup, seed + i//2, max_total_turns=max_total_turns)
        player_to_agent = (1, 0) if swapped else (0, 1)
        if result.winner is None:
            draws += 1
        else:
            agent_index = player_to_agent[result.winner]
            wins[agent_index] += 1
            seat_wins[agent_index][result.winner] += 1
        for seat in (0, 1):
            idx = player_to_agent[seat]
            turns[idx] += result.turns[seat]
            rolls[idx] += result.rolls[seat]
            busts[idx] += result.busts[seat]
            stops[idx] += result.stops[seat]
        if include_game_records:
            a_seat = 1 if swapped else 0
            b_seat = 1 - a_seat
            records.append({
                "game": i, "seed": seed + i//2, "a_seat": a_seat,
                "winner": "draw" if result.winner is None else ("A" if player_to_agent[result.winner] == 0 else "B"),
                "a_turns": result.turns[a_seat], "b_turns": result.turns[b_seat],
                "a_rolls": result.rolls[a_seat], "b_rolls": result.rolls[b_seat],
                "a_busts": result.busts[a_seat], "b_busts": result.busts[b_seat],
                "a_stops": result.stops[a_seat], "b_stops": result.stops[b_seat],
            })
        if progress_callback is not None:
            progress_callback(i + 1, games)
    report = {
        "games": games, "seed": seed, "draws": draws,
        "agents": [
            {
                "id": name, "wins": wins[j], "win_rate": wins[j] / games,
                "win_rate_wilson95": wilson_interval(wins[j], games),
                "seat0_wins": seat_wins[j][0], "seat1_wins": seat_wins[j][1],
                "total_turns": turns[j], "avg_turns_per_game": turns[j]/games,
                "total_rolls": rolls[j], "avg_rolls_per_game": rolls[j]/games,
                "total_busts": busts[j], "busts_per_turn": busts[j]/turns[j] if turns[j] else 0.0,
                "total_stops": stops[j],
            }
            for j, name in enumerate(("A", "B"))
        ],
    }
    if include_game_records:
        report["game_records"] = records
    return report
