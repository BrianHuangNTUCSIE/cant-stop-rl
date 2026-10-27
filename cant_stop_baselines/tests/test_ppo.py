import tempfile
import unittest
from pathlib import Path
import random

import torch

from cant_stop_core.agents import CoutoAgent, Rule28Agent
from cant_stop_core.game import Game
from cant_stop_core.ppo import (
    PPOActorCritic, PPOAgent, initialize_from_ddqn, initialize_from_ppo,
    joint_action_masks, load_checkpoint, mask_policy_logits, save_checkpoint,
)
from evaluate_ml import ModelSpec, load_saved_agent
from train_ppo import (
    collect_pairing_imitation_examples, explained_variance,
    initialize_pairing_by_imitation, pairing_only_parameters,
    rank_validation_checkpoints, restrict_pairing_gradients,
    validation_behavior,
)


class TestPPO(unittest.TestCase):
    def test_explained_variance(self):
        targets = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(explained_variance(targets, targets), 1.0)
        self.assertEqual(
            explained_variance(torch.zeros(3), torch.ones(3)), 0.0
        )

    def test_validation_checkpoints_rank_by_minimum_rate(self):
        entries = [
            {"episodes": 20, "selection_score": 0.6},
            {"episodes": 10, "selection_score": 0.6},
            {"episodes": 30, "selection_score": 0.7},
            {"episodes": 40, "selection_score": 0.5},
        ]
        ranked = rank_validation_checkpoints(entries)
        self.assertEqual([entry["episodes"] for entry in ranked], [30, 10, 20])

    def test_validation_behavior_aggregates_counts(self):
        reports = {
            "first": {"agents": [{
                "total_turns": 10, "total_rolls": 40,
                "total_busts": 10, "total_stops": 20,
            }]},
            "second": {"agents": [{
                "total_turns": 10, "total_rolls": 20,
                "total_busts": 0, "total_stops": 10,
            }]},
        }
        behavior = validation_behavior(reports)
        self.assertEqual(behavior["busts_per_turn"], 0.5)
        self.assertEqual(behavior["stops_per_turn"], 1.5)
        self.assertEqual(behavior["rolls_per_turn"], 3.0)
        self.assertEqual(behavior["greedy_stop_rate"], 0.6)

    def test_forward_shapes(self):
        model = PPOActorCritic(68, (128, 128))
        logits, values = model(torch.zeros((4, 68)))
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(tuple(values.shape), (4,))

        joint = PPOActorCritic(86, (128, 128), 8)
        logits, values = joint(torch.zeros((4, 86)))
        self.assertEqual(tuple(logits.shape), (4, 8))
        self.assertEqual(tuple(values.shape), (4,))

    def test_joint_masks_exclude_wrong_phase_actions(self):
        pair_observation = torch.zeros((1, 86))
        pair_observation[0, 70] = 1.0
        pair_observation[0, 73] = 1.0
        decision_observation = torch.zeros((1, 86))
        masks = joint_action_masks(torch.cat((pair_observation, decision_observation)))
        self.assertEqual(
            masks.tolist(),
            [[True, True, False, False, False, False, False, False],
             [False, False, False, False, False, False, True, True]],
        )
        masked = mask_policy_logits(torch.zeros((2, 8)), masks)
        self.assertTrue(torch.isneginf(masked[0, 6]))
        self.assertTrue(torch.isneginf(masked[1, 0]))

    def test_initialize_from_reference_ddqn(self):
        model = PPOActorCritic(68, (128, 128))
        initialize_from_ddqn(
            model,
            Path("runs/ddqn_mc_counterfactual_seed447/best.pt"),
        )
        logits, _ = model(torch.zeros((1, 68)))
        self.assertEqual(tuple(logits.shape), (1, 2))

    def test_checkpoint_round_trip(self):
        model = PPOActorCritic(68, (16, 16))
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.pt"
            save_checkpoint(path, model, optimizer, {
                "observation_size": 68, "hidden_sizes": (16, 16),
            }, episodes=3, updates=2)
            loaded, config = load_checkpoint(path)
            self.assertEqual(config["observation_size"], 68)
            self.assertEqual(tuple(config["hidden_sizes"]), (16, 16))
            self.assertEqual(
                model(torch.zeros((1, 68)))[0].shape,
                loaded(torch.zeros((1, 68)))[0].shape,
            )

    def test_joint_checkpoint_round_trip_and_ppo_transfer(self):
        source = PPOActorCritic(68, (16, 16))
        source_optimizer = torch.optim.Adam(source.parameters())
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.pt"
            save_checkpoint(source_path, source, source_optimizer, {
                "algorithm": "ppo", "observation_size": 68,
                "action_size": 2, "hidden_sizes": (16, 16),
            }, episodes=3, updates=2)
            joint = PPOActorCritic(86, (16, 16), 8)
            initialize_from_ppo(joint, source_path)
            self.assertTrue(torch.equal(
                joint.trunk[0].weight[:, :68], source.trunk[0].weight
            ))
            self.assertEqual(
                float(joint.trunk[0].weight[:, 68:].detach().abs().sum()), 0.0
            )
            self.assertTrue(torch.equal(
                joint.policy_head.weight[6:8], source.policy_head.weight
            ))
            joint_path = Path(directory) / "joint.pt"
            save_checkpoint(joint_path, joint, torch.optim.Adam(joint.parameters()), {
                "algorithm": "ppo", "observation_size": 86,
                "action_size": 8, "hidden_sizes": (16, 16),
                "pairing_policy": "learned",
            }, episodes=4, updates=3)
            loaded, _ = load_checkpoint(joint_path)
            self.assertEqual(loaded.action_size, 8)
            self.assertEqual(tuple(loaded(torch.zeros((1, 86)))[0].shape), (1, 8))

    def test_pairing_imitation_collection_is_seeded(self):
        first = collect_pairing_imitation_examples(16, 123)
        second = collect_pairing_imitation_examples(16, 123)
        for first_tensor, second_tensor in zip(first, second):
            self.assertTrue(torch.equal(first_tensor, second_tensor))

    def test_pairing_imitation_preserves_continue_stop_rows(self):
        model = PPOActorCritic(86, (16, 16), 8)
        before_state_columns = model.trunk[0].weight[:, :68].detach().clone()
        before_weight = model.policy_head.weight[6:8].detach().clone()
        before_bias = model.policy_head.bias[6:8].detach().clone()
        metrics = initialize_pairing_by_imitation(
            model, examples=32, epochs=1, seed=19, minibatch_size=16,
        )
        self.assertTrue(torch.equal(before_weight, model.policy_head.weight[6:8]))
        self.assertTrue(torch.equal(before_bias, model.policy_head.bias[6:8]))
        self.assertTrue(torch.equal(
            before_state_columns, model.trunk[0].weight[:, :68]
        ))
        self.assertGreater(
            float(model.trunk[0].weight[:, 68:].detach().abs().sum()), 0.0
        )
        self.assertGreaterEqual(metrics["accuracy"], 0.0)

    def test_pairing_only_gradient_mask_preserves_decision_parameters(self):
        model = PPOActorCritic(86, (16, 16), 8)
        loss = model(torch.randn((4, 86)))[0].sum()
        loss.backward()
        first = model.trunk[0].weight
        state_gradient = first.grad[:, :68].detach().clone()
        decision_gradient = model.policy_head.weight.grad[6:8].detach().clone()
        restrict_pairing_gradients(model)
        self.assertEqual(float(first.grad[:, :68].abs().sum()), 0.0)
        self.assertEqual(float(model.policy_head.weight.grad[6:8].abs().sum()), 0.0)
        self.assertGreater(float(state_gradient.abs().sum()), 0.0)
        self.assertGreater(float(decision_gradient.abs().sum()), 0.0)
        self.assertEqual(len(pairing_only_parameters(model)), 3)

    def test_joint_agent_selects_only_legal_pairing(self):
        model = PPOActorCritic(86, (16, 16), 8)
        agent = PPOAgent(model)
        game = Game()
        moves = game.roll(random.Random(1), dice=(1, 2, 3, 4))
        selected = agent.choose_move(game.snapshot(), moves, random.Random(2))
        self.assertIn(selected, moves)

    def test_evaluator_overrides_frozen_ppo_pairing(self):
        model = PPOActorCritic(68, (16, 16))
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.pt"
            save_checkpoint(path, model, optimizer, {
                "algorithm": "ppo", "observation_size": 68,
                "hidden_sizes": (16, 16),
            }, episodes=3, updates=2)
            ga_agent, _ = load_saved_agent(
                ModelSpec("ga", path, "ga"), "cpu"
            )
            couto_agent, _ = load_saved_agent(
                ModelSpec("couto", path, "couto"), "cpu"
            )
            self.assertIsInstance(ga_agent.pairing_policy, Rule28Agent)
            self.assertIsInstance(couto_agent.pairing_policy, CoutoAgent)


if __name__ == "__main__":
    unittest.main()
