"""Full-game Monte Carlo labels for value-policy distillation.

This module deliberately uses the existing game engine and pairing policy.  A
label is the estimated probability of eventually winning after forcing STOP or
CONTINUE at a recorded decision state.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .agents import Agent, MonteCarloAgent, Rule28Agent
from .game import Game
from .model import State


@dataclass(frozen=True)
class RolloutConfig:
    rollouts: int = 32
    opponent_rollouts: int = 8
    future_roll_limit: int = 16
    max_turns: int = 200


def _game_from_state(state: State) -> Game:
    game = Game()
    game.permanent = [list(state.permanent[0]), list(state.permanent[1])]
    game.owner = list(state.owner)
    game.active = state.active_dict()
    game.current_player = state.current_player
    game.rolls_this_turn = state.rolls_this_turn
    game.phase = "decision"
    return game


def _mc_policy(config: RolloutConfig) -> Agent:
    return MonteCarloAgent(
        rollouts=config.opponent_rollouts,
        future_roll_limit=config.future_roll_limit,
    )


def _finish_game(game: Game, *, learner: int, config: RolloutConfig,
                 rng: random.Random, policy: Agent, pairing: Agent) -> float:
    """Continue a cloned game and return a learner-relative win indicator."""
    while game.winner is None:
        if game.phase == "roll" and game.rolls_this_turn == 0:
            if sum(game.turns) >= config.max_turns:
                return 0.0
        player = game.current_player
        if game.phase == "roll":
            if not game.roll(rng):
                continue
        if game.phase == "pair":
            move = pairing.choose_move(game.snapshot(), game.available, rng)
            game.apply_move(move)
        if game.phase == "decision":
            if policy.should_stop(game.snapshot(), rng):
                game.stop()
            else:
                game.continue_turn()
    return float(game.winner == learner)


def action_win_probability(state: State, action: int, *, config: RolloutConfig,
                           seed: int) -> float:
    """Estimate P(win | action) using complete-game rollouts."""
    if action not in (0, 1):
        raise ValueError("action must be 0 (continue) or 1 (stop)")
    wins = 0.0
    learner = state.current_player
    # These policies are stateless; construct them once per label rather than
    # once per rollout.  Keeping them outside the rollout loop does not alter
    # any RNG calls because all randomness remains owned by ``rng``.
    policy = _mc_policy(config)
    pairing = Rule28Agent(pair_heuristic="ga", smart=True)
    for index in range(config.rollouts):
        rng = random.Random(seed + index * 0x9E3779B1)
        game = _game_from_state(state)
        if action:
            game.stop()
        else:
            game.continue_turn()
        wins += _finish_game(
            game, learner=learner, config=config, rng=rng,
            policy=policy, pairing=pairing,
        )
    return wins / config.rollouts


def label_state(state: State, *, config: RolloutConfig, seed: int) -> tuple[float, float]:
    """Return (continue_value, stop_value) for one state."""
    return (
        action_win_probability(state, 0, config=config, seed=seed),
        action_win_probability(state, 1, config=config, seed=seed ^ 0xA5A5A5A5),
    )
