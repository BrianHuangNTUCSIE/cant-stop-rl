"""Dependency-free RL environments for Can't Stop policies."""
from __future__ import annotations

import random
from enum import IntEnum

from .agents import Agent, Rule28Agent
from .constants import LENGTHS
from .game import Game


class StopContinueAction(IntEnum):
    """Actions available after a successful roll and pairing choice."""

    CONTINUE = 0
    STOP = 1


class JointAction(IntEnum):
    """Fixed action layout for learned pairing plus Stop/Continue.

    Actions 0--5 select the corresponding sorted legal pairing.  Actions 6
    and 7 are Continue and Stop respectively.
    """

    CONTINUE = 6
    STOP = 7


class StopContinueEnv:
    """Two-player environment exposing only one player's banking decisions.

    Dice pairing is handled by a fixed policy for both players. The opponent's
    complete turns are also automated. Consequently, ``step`` advances from one
    learner decision to the next learner decision (or the end of the game).

    The API follows Gymnasium's reset/step result shapes without depending on
    Gymnasium. Observations are always from the learner's perspective because
    they are emitted only while it is the learner's turn.
    """

    CONTINUE = StopContinueAction.CONTINUE
    STOP = StopContinueAction.STOP

    def __init__(
        self,
        *,
        enhanced_features: bool = False,
        domain_features: bool = False,
        strategic_features: bool = False,
        learner_player: int = 0,
        pairing_policy: Agent | None = None,
        opponent: Agent | None = None,
        seed: int = 0,
        max_total_turns: int = 2000,
        reward_shaping: float = 0.0,
        shaping_gamma: float = 0.99,
        shaping_profile: str = "progress",
        learn_pairing: bool = False,
    ) -> None:
        if learner_player not in (0, 1):
            raise ValueError("learner_player must be 0 or 1")
        if max_total_turns < 1:
            raise ValueError("max_total_turns must be >= 1")
        self.enhanced_features = enhanced_features
        self.domain_features = domain_features
        self.strategic_features = strategic_features
        self.learner_player = learner_player
        self.pairing_policy = pairing_policy or Rule28Agent(
            pair_heuristic="ga", smart=True
        )
        self.opponent = opponent or Rule28Agent(pair_heuristic="ga", smart=True)
        self.seed = seed
        self.max_total_turns = max_total_turns
        if reward_shaping < 0.0:
            raise ValueError("reward_shaping must be non-negative")
        if not 0.0 <= shaping_gamma <= 1.0:
            raise ValueError("shaping_gamma must be in [0, 1]")
        if shaping_profile not in ("progress", "race-aware"):
            raise ValueError("shaping_profile must be 'progress' or 'race-aware'")
        self.reward_shaping = reward_shaping
        self.shaping_gamma = shaping_gamma
        self.shaping_profile = shaping_profile
        self.learn_pairing = learn_pairing
        self.game = Game()
        self._truncated = False
        self._seed_rngs(seed)

    @property
    def observation_size(self) -> int:
        base = (80 if self.strategic_features else
                (68 if self.domain_features else
                 (60 if self.enhanced_features else 58)))
        return base + (18 if self.learn_pairing else 0)

    @property
    def action_size(self) -> int:
        return 8 if self.learn_pairing else 2

    @staticmethod
    def _move_features(moves) -> list[float]:
        """Encode six canonical pairing slots as (first, second, valid)."""
        values: list[float] = []
        for move in sorted(moves)[:6]:
            columns = move.columns
            values.extend((columns[0] / 12.0,
                           columns[1] / 12.0 if len(columns) == 2 else 0.0,
                           1.0))
        values.extend([0.0] * (18 - len(values)))
        return values

    def _seed_rngs(self, seed: int) -> None:
        # Keep environment chance and each physical seat's policy randomness
        # separate so changing a policy does not alter the dice sequence.
        self._dice_rng = random.Random(seed)
        self._policy_rng = (
            random.Random(seed ^ 0x1234ABCD),
            random.Random(seed ^ 0xFEDC5678),
        )

    def _observation(self) -> list[float]:
        if (self.game.phase not in ("decision", "pair")
                or self.game.current_player != self.learner_player):
            raise RuntimeError("An observation is only available at a learner action")
        features = self.game.snapshot().features(
            enhanced=self.enhanced_features, domain=self.domain_features,
            strategic=self.strategic_features,
        )
        if self.learn_pairing:
            features += self._move_features(
                self.game.available if self.game.phase == "pair" else ()
            )
        return features

    def legal_action_mask(self) -> list[bool]:
        """Return legal actions in the fixed action layout."""
        if not self.learn_pairing:
            return [True, True]
        if self.game.phase == "pair":
            return [index < len(self.game.available) for index in range(6)] + [False, False]
        if self.game.phase == "decision":
            return [False] * 6 + [True, True]
        return [False] * self.action_size

    def _potential(self) -> float:
        """Bounded learner-relative potential used for policy-safe shaping.

        ``progress`` is the original potential. ``race-aware`` additionally
        values completion threats and discounts temporary progress, which is
        useful for learning against race-sensitive opponents such as MC.
        """
        state = self.game.snapshot()
        learner = self.learner_player
        opponent = 1 - learner
        completed_difference = (
            state.completed(learner) - state.completed(opponent)
        ) / 3.0
        progress_difference = sum(
            (state.permanent[learner][i] - state.permanent[opponent][i])
            / LENGTHS[i]
            for i in range(11)
        ) / 11.0
        if self.shaping_profile == "progress":
            return completed_difference + 0.35 * progress_difference

        learner_progress = state.permanent[learner]
        opponent_progress = state.permanent[opponent]
        learner_threat = sum(
            owner == -1 and LENGTHS[i] - opponent_progress[i] <= 1
            for i, owner in enumerate(state.owner)
        )
        opponent_threat = sum(
            owner == -1 and LENGTHS[i] - learner_progress[i] <= 1
            for i, owner in enumerate(state.owner)
        )
        active = state.active_dict()
        unbanked = sum(
            max(0, position - learner_progress[column - 2]) / LENGTHS[column - 2]
            for column, position in active.items()
        ) / 3.0
        return (
            completed_difference
            + 0.35 * progress_difference
            + 0.10 * (opponent_threat - learner_threat) / 3.0
            - 0.10 * unbanked
        )

    def _info(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "learner_player": self.learner_player,
            "winner": self.game.winner,
            "turns": tuple(self.game.turns),
            "rolls": tuple(self.game.rolls),
            "busts": tuple(self.game.busts),
            "stops": tuple(self.game.stops),
        }

    def _advance_to_learner_decision(self) -> None:
        while self.game.winner is None:
            # As in the evaluator, finish a started turn but do not begin a new
            # turn after the configured safety limit.
            if (self.game.phase == "roll" and self.game.rolls_this_turn == 0
                    and sum(self.game.turns) >= self.max_total_turns):
                self._truncated = True
                return

            player = self.game.current_player
            if self.game.phase == "roll":
                moves = self.game.roll(self._dice_rng)
                if not moves:  # Bust ends the turn inside Game.roll().
                    continue

            if self.game.phase == "pair":
                if player == self.learner_player and self.learn_pairing:
                    return
                state = self.game.snapshot()
                policy = (self.pairing_policy if player == self.learner_player
                          else self.opponent)
                move = policy.choose_move(
                    state, self.game.available, self._policy_rng[player]
                )
                self.game.apply_move(move)

            if self.game.phase == "decision":
                if player == self.learner_player and self.learn_pairing:
                    return
                if player == self.learner_player:
                    return
                state = self.game.snapshot()
                if self.opponent.should_stop(state, self._policy_rng[player]):
                    self.game.stop()
                else:
                    self.game.continue_turn()

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[list[float] | None, dict[str, object]]:
        """Start a fresh seeded game and advance to the first learner decision."""
        if seed is not None:
            self.seed = seed
        self._seed_rngs(self.seed)
        self.game = Game()
        self._truncated = False
        self._advance_to_learner_decision()
        observation = (None if self.game.winner is not None or self._truncated
                       else self._observation())
        return observation, self._info()

    def step(
        self, action: int | StopContinueAction
    ) -> tuple[list[float] | None, float, bool, bool, dict[str, object]]:
        """Apply a learner action and run until its next decision or game end."""
        if self.game.winner is not None or self._truncated:
            raise RuntimeError("step() called after the episode ended; call reset()")
        if (self.game.phase not in ("decision", "pair")
                or self.game.current_player != self.learner_player):
            raise RuntimeError("step() is only allowed at a learner action")
        try:
            if self.learn_pairing:
                action = int(action)
                if action < 0 or action >= self.action_size or not self.legal_action_mask()[action]:
                    raise ValueError
                if self.game.phase == "pair":
                    previous_potential = self._potential()
                    self.game.apply_move(sorted(self.game.available)[action])
                    self._advance_to_learner_decision()
                    observation = self._observation()
                    next_potential = (0.0 if self._truncated else self._potential())
                    shaping_reward = self.reward_shaping * (
                        self.shaping_gamma * next_potential - previous_potential
                    )
                    info = self._info()
                    info.update({
                        "extrinsic_reward": 0.0,
                        "shaping_reward": shaping_reward,
                    })
                    return observation, shaping_reward, False, self._truncated, info
                selected = JointAction(action)
                stop = selected is JointAction.STOP
            else:
                selected = StopContinueAction(action)
                stop = selected is StopContinueAction.STOP
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("invalid action for the current learner phase") from exc

        previous_potential = self._potential()
        if stop:
            self.game.stop()
        else:
            self.game.continue_turn()
        self._advance_to_learner_decision()

        terminated = self.game.winner is not None
        extrinsic_reward = 0.0
        if terminated:
            extrinsic_reward = (
                1.0 if self.game.winner == self.learner_player else -1.0
            )
        next_potential = 0.0 if terminated or self._truncated else self._potential()
        shaping_reward = self.reward_shaping * (
            self.shaping_gamma * next_potential - previous_potential
        )
        reward = extrinsic_reward + shaping_reward
        observation = None if terminated or self._truncated else self._observation()
        info = self._info()
        info.update({
            "extrinsic_reward": extrinsic_reward,
            "shaping_reward": shaping_reward,
        })
        return observation, reward, terminated, self._truncated, info
