import argparse
import random
import unittest

from cant_stop_core.classical_ml import ClassicalMLAgent
from cant_stop_core.game import Game
from cant_stop_core.rl_env import StopContinueAction
from generate_ml_dataset import (branch_outcome, game_from_decision_state,
                                 label_state)
from evaluate_ml import (
    paired_difference_interval, parse_model_spec, parse_ppo_pairing,
)


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        return [self.value] * len(rows)


class TestClassicalMLSupport(unittest.TestCase):
    def test_evaluator_model_spec_supports_explicit_checkpoint_names(self):
        explicit = parse_model_spec("ppo-best=runs/example/best.pt")
        implicit = parse_model_spec("runs/example/model.joblib")
        self.assertEqual(explicit.name, "ppo-best")
        self.assertEqual(str(explicit.path), "runs/example/best.pt")
        self.assertEqual(implicit.name, "model")

    def test_evaluator_parses_ppo_pairing_override(self):
        self.assertEqual(parse_ppo_pairing("candidate=couto"), ("candidate", "couto"))
        self.assertEqual(parse_ppo_pairing("control=ga"), ("control", "ga"))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_ppo_pairing("candidate=random")

    @staticmethod
    def decision_state():
        game = Game()
        moves = game.roll(random.Random(1), dice=(1, 2, 3, 4))
        game.apply_move(moves[0])
        return game.snapshot()

    def test_reconstructs_independent_decision_game(self):
        state = self.decision_state()
        restored = game_from_decision_state(state)
        restored.stop()
        self.assertEqual(restored.phase, "roll")
        self.assertEqual(state.active, self.decision_state().active)

    def test_paired_rollout_label_is_deterministic_and_bounded(self):
        state = self.decision_state()
        first = label_state(state, rollouts=2, seed=991, max_future_turns=200)
        second = label_state(state, rollouts=2, seed=991, max_future_turns=200)
        self.assertEqual(first, second)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in first))
        outcome = branch_outcome(
            state, StopContinueAction.STOP, seed=123, max_future_turns=200
        )
        self.assertIn(outcome, (0.0, 0.5, 1.0))

    def test_agent_supports_action_and_margin_targets(self):
        state = self.decision_state()
        stop_agent = ClassicalMLAgent(
            ConstantModel(1), feature_mode="raw", prediction_target="action"
        )
        continue_agent = ClassicalMLAgent(
            ConstantModel(0.2), feature_mode="domain",
            prediction_target="continue_margin",
        )
        self.assertTrue(stop_agent.should_stop(state, random.Random(1)))
        self.assertFalse(continue_agent.should_stop(state, random.Random(1)))

    def test_paired_model_difference_uses_shared_seed_units(self):
        estimate, low, high = paired_difference_interval(
            {1: 1.0, 2: 0.5, 3: 1.0},
            {1: 0.0, 2: 0.5, 3: 0.0},
            samples=200, seed=3,
        )
        self.assertAlmostEqual(estimate, 2.0 / 3.0)
        self.assertLessEqual(low, estimate)
        self.assertGreaterEqual(high, estimate)


if __name__ == "__main__":
    unittest.main()
