import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch RL extra is not installed")
class TestDQNComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cant_stop_core import dqn
        cls.dqn = dqn

    def test_auto_device_returns_available_backend(self):
        selected = self.dqn.select_device("auto")
        self.assertIn(selected.type, self.dqn.available_device_types())

    def test_network_and_replay_shapes(self):
        device = self.dqn.select_device("cpu")
        model = self.dqn.QNetwork(4, 2, (8,))
        output = model(torch.zeros((3, 4)))
        self.assertEqual(tuple(output.shape), (3, 2))

        replay = self.dqn.ReplayBuffer(10, 4, random.Random(1))
        replay.push([1, 2, 3, 4], 1, 0.5, [2, 3, 4, 5], False)
        replay.push([2, 3, 4, 5], 0, -1.0, None, True)
        batch = replay.sample(2, device)
        self.assertEqual(tuple(batch[0].shape), (2, 4))
        self.assertEqual(tuple(batch[1].shape), (2,))
        self.assertEqual(int(batch[4].sum().item()), 1)

    def test_dueling_network_shape(self):
        model = self.dqn.QNetwork(80, 2, (32, 16), "dueling")
        output = model(torch.zeros((3, 80)))
        self.assertEqual(tuple(output.shape), (3, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_replay_batch_insert_wraps_without_copying_capacity(self):
        device = self.dqn.select_device("cpu")
        replay = self.dqn.ReplayBuffer(3, 2, random.Random(3))
        replay.push_batch(
            [[0, 0], [1, 1], [2, 2], [3, 3]],
            [0, 1, 0, 1], [0, 0, 0, 1],
            [[1, 1], [2, 2], [3, 3], None],
            [False, False, False, True],
        )
        states, _, _, _, dones = replay.sample(3, device)
        self.assertEqual(len(replay), 3)
        self.assertEqual(sorted(states[:, 0].tolist()), [1.0, 2.0, 3.0])
        self.assertEqual(int(dones.sum().item()), 1)

    def test_versioned_transition_batch_inserts_into_replay(self):
        from cant_stop_core.transition import Transition, TransitionBatch

        device = self.dqn.select_device("cpu")
        replay = self.dqn.ReplayBuffer(3, 2, random.Random(3))
        transition = Transition.create(
            [1, 2], 1, 0.5, None, True, observation_size=2,
            episode_id=7, worker_id=2, step_id=4,
        )
        replay.push_transition_batch(TransitionBatch.from_iterable(2, [transition]))
        states, actions, rewards, next_states, dones = replay.sample(1, device)
        self.assertEqual(states.tolist(), [[1.0, 2.0]])
        self.assertEqual(actions.tolist(), [1])
        self.assertEqual(rewards.tolist(), [0.5])
        self.assertEqual(next_states.tolist(), [[0.0, 0.0]])
        self.assertEqual(dones.tolist(), [True])

    def test_batched_greedy_action_uses_one_policy(self):
        device = self.dqn.select_device("cpu")
        model = self.dqn.QNetwork(4, 2, (8,)).to(device)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.network[-1].bias[1] = 1.0
        actions = self.dqn.choose_actions(
            model, [[0, 0, 0, 0]] * 5, 0.0, random.Random(4), device
        )
        self.assertEqual(actions, [1] * 5)

    def test_dqn_and_double_dqn_updates(self):
        device = self.dqn.select_device("cpu")
        for algorithm in ("dqn", "double-dqn"):
            config = self.dqn.DQNConfig(
                observation_size=4, hidden_sizes=(8,), algorithm=algorithm,
                batch_size=2, replay_capacity=10,
            )
            online = self.dqn.QNetwork(4, 2, (8,)).to(device)
            target = self.dqn.QNetwork(4, 2, (8,)).to(device)
            target.load_state_dict(online.state_dict())
            optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
            replay = self.dqn.ReplayBuffer(10, 4, random.Random(2))
            replay.push([0, 0, 0, 0], 0, 0, [1, 0, 0, 0], False)
            replay.push([1, 0, 0, 0], 1, 1, None, True)
            loss = self.dqn.optimize_batch(
                online, target, optimizer, replay, config, device
            )
            self.assertGreaterEqual(float(loss.cpu().item()), 0.0)

    def test_learner_optimizes_without_environment_dependency(self):
        from learners.dqn_learner import DQNLearner

        device = self.dqn.select_device("cpu")
        config = self.dqn.DQNConfig(
            observation_size=2, hidden_sizes=(4,), batch_size=2,
            replay_capacity=4, warmup_steps=0, train_frequency=1,
        )
        online = self.dqn.QNetwork(2, 2, (4,)).to(device)
        target = self.dqn.QNetwork(2, 2, (4,)).to(device)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        replay = self.dqn.ReplayBuffer(4, 2, random.Random(2))
        replay.push([0, 0], 0, 0, [1, 0], False)
        replay.push([1, 0], 1, 1, None, True)
        learner = DQNLearner(
            online, target, optimizer, replay, config, device
        )
        updates, losses = learner.optimize_due(1, 0)
        self.assertEqual(updates, 2)
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses))

    def test_counterfactual_replay_and_update(self):
        device = self.dqn.select_device("cpu")
        config = self.dqn.DQNConfig(
            observation_size=4, hidden_sizes=(8,), batch_size=2,
            replay_capacity=10, counterfactual_loss_weight=0.1,
        )
        online = self.dqn.QNetwork(4, 2, (8,)).to(device)
        target = self.dqn.QNetwork(4, 2, (8,)).to(device)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        replay = self.dqn.ReplayBuffer(10, 4, random.Random(8))
        replay.push(
            [0, 0, 0, 0], 0, 0, [1, 0, 0, 0], False,
            counterfactual=(0.8, 0.2), counterfactual_mask=True,
        )
        replay.push(
            [1, 0, 0, 0], 1, 1, None, True,
            counterfactual=(0.2, 0.7), counterfactual_mask=True,
        )
        sampled = replay.sample(
            2, device, include_n_steps=True, include_counterfactual=True,
        )
        self.assertEqual(tuple(sampled[6].shape), (2, 2))
        self.assertEqual(tuple(sampled[7].shape), (2,))
        loss = self.dqn.optimize_batch(
            online, target, optimizer, replay, config, device
        )
        self.assertTrue(torch.isfinite(loss))

    def test_joint_target_mask_matches_pair_and_decision_phases(self):
        device = self.dqn.select_device("cpu")
        observations = torch.zeros((2, 76), device=device)
        # First row is a pairing state with only slots 0 and 2 legal.
        observations[0, 58 + 2] = 1.0
        observations[0, 58 + 8] = 1.0
        masks = self.dqn.joint_legal_action_mask(observations)
        self.assertEqual(masks[0].tolist(),
                         [True, False, True, False, False, False, False, False])
        self.assertEqual(masks[1].tolist(),
                         [False, False, False, False, False, False, True, True])

    def test_prioritized_replay_samples_and_updates_priorities(self):
        device = self.dqn.select_device("cpu")
        replay = self.dqn.PrioritizedReplayBuffer(
            4, 2, random.Random(7), alpha=0.6, beta=0.4,
        )
        replay.push_batch(
            [[0, 0], [1, 1], [2, 2], [3, 3]],
            [0, 1, 0, 1], [0, 0, 0, 1],
            [[1, 1], [2, 2], [3, 3], None],
            [False, False, False, True],
        )
        sampled = replay.sample(2, device, include_n_steps=True,
                                include_priorities=True)
        self.assertEqual(tuple(sampled[0].shape), (2, 2))
        self.assertEqual(tuple(sampled[-2].shape), (2,))
        self.assertEqual(tuple(sampled[-1].shape), (2,))
        replay.update_priorities(sampled[-1], torch.tensor([2.0, 3.0]))
        self.assertGreaterEqual(replay._max_priority, 3.0)

    def test_n_step_accumulator_discounts_and_flushes(self):
        accumulator = self.dqn.NStepAccumulator(3, 0.5)
        self.assertEqual(accumulator.append([0], 0, 1.0, [1], False), [])
        self.assertEqual(accumulator.append([1], 1, 2.0, [2], False), [])
        emitted = accumulator.append([2], 0, 4.0, [3], False)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].reward, 3.0)
        self.assertEqual(emitted[0].n_steps, 3)
        self.assertFalse(emitted[0].done)

        emitted = accumulator.append([3], 1, 8.0, None, True)
        self.assertEqual([item.n_steps for item in emitted], [3, 2, 1])
        self.assertTrue(all(item.done for item in emitted))
        self.assertEqual([item.reward for item in emitted], [6.0, 8.0, 8.0])

    def test_checkpoint_is_portable_and_loads_as_agent(self):
        device = self.dqn.select_device("cpu")
        config = self.dqn.DQNConfig(hidden_sizes=(8,), batch_size=2)
        online = self.dqn.QNetwork(58, 2, (8,)).to(device)
        target = self.dqn.QNetwork(58, 2, (8,)).to(device)
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.dqn.save_checkpoint(
                path, online, target, optimizer, config,
                global_step=12, episodes=3,
                metadata={"torch": str(torch.__version__), "device": "cpu"},
            )
            checkpoint = self.dqn.load_checkpoint(path, device)
            agent = self.dqn.DQNAgent.from_checkpoint(path, device="cpu")
            self.assertEqual(checkpoint["global_step"], 12)
            self.assertFalse(agent.enhanced_features)

    def test_strategic_dueling_checkpoint_round_trip(self):
        device = self.dqn.select_device("cpu")
        config = self.dqn.DQNConfig(
            observation_size=80, hidden_sizes=(16, 8), batch_size=2,
            algorithm="double-dqn", network_type="dueling",
            pairing_policy="couto",
        )
        online = self.dqn.QNetwork(80, 2, (16, 8), "dueling")
        target = self.dqn.QNetwork(80, 2, (16, 8), "dueling")
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.dqn.save_checkpoint(
                path, online, target, optimizer, config,
                global_step=1, episodes=1,
            )
            agent = self.dqn.DQNAgent.from_checkpoint(path, device="cpu")
            self.assertTrue(agent.strategic_features)
            self.assertEqual(agent.model.network_type, "dueling")

    def test_strategic_learned_pairing_checkpoint_round_trip(self):
        device = self.dqn.select_device("cpu")
        config = self.dqn.DQNConfig(
            observation_size=98, action_size=8, hidden_sizes=(16,), batch_size=2,
            algorithm="double-dqn", pairing_policy="learned",
        )
        online = self.dqn.QNetwork(98, 8, (16,))
        target = self.dqn.QNetwork(98, 8, (16,))
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.dqn.save_checkpoint(
                path, online, target, optimizer, config,
                global_step=1, episodes=1,
            )
            agent = self.dqn.DQNAgent.from_checkpoint(path, device="cpu")
            self.assertTrue(agent.strategic_features)
            from cant_stop_core.rl_env import StopContinueEnv
            env = StopContinueEnv(
                strategic_features=True, learn_pairing=True, seed=7,
            )
            env.reset()
            self.assertEqual(
                len(agent._features(env.game.snapshot(), env.game.available)), 98
            )

    def test_validation_behavior_aggregates_greedy_policy_statistics(self):
        from train_dqn import validation_behavior

        reports = {
            "a": {"agents": [{
                "total_turns": 10, "total_rolls": 30,
                "total_busts": 5, "total_stops": 10,
            }]},
            "b": {"agents": [{
                "total_turns": 20, "total_rolls": 60,
                "total_busts": 10, "total_stops": 20,
            }]},
        }
        behavior = validation_behavior(reports)
        self.assertEqual(behavior["busts_per_turn"], 0.5)
        self.assertEqual(behavior["stops_per_turn"], 1.0)
        self.assertEqual(behavior["rolls_per_turn"], 3.0)
        self.assertEqual(behavior["greedy_stop_rate"], 0.4)

    def test_parallel_validation_preserves_seeded_reports(self):
        from train_dqn import validation_reports

        config = self.dqn.DQNConfig(
            observation_size=58, action_size=2, hidden_sizes=(8,), batch_size=2,
        )
        model = self.dqn.QNetwork(58, 2, (8,))
        kwargs = dict(
            games=4, seed=123, max_turns=20,
            enhanced_features=False, domain_features=False,
            strategic_features=False, opponents=("random", "rule28"),
            mc_rollouts=1, seed_blocks=2, include_records=True,
        )
        sequential = validation_reports(
            model, config, validation_workers=0, **kwargs,
        )
        parallel = validation_reports(
            model, config, validation_workers=2, **kwargs,
        )
        self.assertEqual(parallel, sequential)

    def test_mc_heavy_curriculum_has_three_normalized_stages(self):
        from train_dqn import mc_heavy_curriculum_mix

        stages = [mc_heavy_curriculum_mix(value) for value in (0.0, 0.25, 0.65)]
        self.assertEqual([dict(stage)["mc"] for stage in stages], [0.30, 0.55, 0.65])
        self.assertTrue(all(abs(sum(dict(stage).values()) - 1.0) < 1e-12
                            for stage in stages))


if __name__ == "__main__":
    unittest.main()
