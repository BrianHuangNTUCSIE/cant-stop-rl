import unittest

from cant_stop import JointAction, StopContinueAction, StopContinueEnv


class TestStopContinueEnv(unittest.TestCase):
    def test_reset_reaches_learner_decision(self):
        env = StopContinueEnv(seed=441)

        observation, info = env.reset()

        self.assertEqual(len(observation), 58)
        self.assertEqual(env.game.phase, "decision")
        self.assertEqual(env.game.current_player, env.learner_player)
        self.assertEqual(info["seed"], 441)

    def test_stop_banks_progress_and_returns_to_learner(self):
        env = StopContinueEnv(seed=17)
        env.reset()
        banked = dict(env.game.active)

        observation, reward, terminated, truncated, _ = env.step(
            StopContinueAction.STOP
        )

        for column, position in banked.items():
            self.assertEqual(env.game.permanent[0][column - 2], position)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(len(observation), 58)
        self.assertEqual(env.game.current_player, 0)
        self.assertEqual(env.game.phase, "decision")

    def test_continue_eventually_handles_a_bust_and_switches_back(self):
        # Continuing first fills the neutral markers; a later failed roll then
        # executes the opponent turn and returns at player 0's next decision.
        env = StopContinueEnv(seed=0)
        env.reset()
        busts_before = env.game.busts[0]

        for _ in range(100):
            observation, reward, terminated, truncated, info = env.step(
                StopContinueAction.CONTINUE
            )
            if info["busts"][0] > busts_before or terminated or truncated:
                break

        self.assertGreater(info["busts"][0], busts_before)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNotNone(observation)
        self.assertEqual(env.game.current_player, 0)

    def test_terminal_reward_matches_winner(self):
        env = StopContinueEnv(seed=29)
        observation, _ = env.reset()
        reward = 0.0
        terminated = truncated = False

        # Banking immediately is deliberately simple; this is an environment
        # integration test rather than a claim about policy quality.
        while not (terminated or truncated):
            self.assertIsNotNone(observation)
            observation, reward, terminated, truncated, info = env.step(
                StopContinueAction.STOP
            )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        expected = 1.0 if info["winner"] == env.learner_player else -1.0
        self.assertEqual(reward, expected)
        self.assertIsNone(observation)

    def test_same_seed_reproduces_trajectory(self):
        def trajectory(seed):
            env = StopContinueEnv(seed=seed)
            observation, _ = env.reset()
            records = [(tuple(observation), tuple(env.game.last_dice))]
            for action in (0, 1, 0, 1, 1, 0):
                observation, reward, terminated, truncated, info = env.step(action)
                records.append((
                    None if observation is None else tuple(observation),
                    reward,
                    terminated,
                    truncated,
                    info["winner"],
                    info["turns"],
                    info["rolls"],
                    info["busts"],
                ))
                if terminated or truncated:
                    break
            return records

        self.assertEqual(trajectory(12345), trajectory(12345))

    def test_raw_and_enhanced_observation_dimensions(self):
        raw, _ = StopContinueEnv(seed=7).reset()
        enhanced_env = StopContinueEnv(enhanced_features=True, seed=7)
        enhanced, _ = enhanced_env.reset()

        self.assertEqual(len(raw), 58)
        self.assertEqual(len(enhanced), 60)
        self.assertEqual(enhanced_env.observation_size, 60)
        self.assertEqual(enhanced_env.action_size, 2)

        domain_env = StopContinueEnv(domain_features=True, seed=7)
        domain, _ = domain_env.reset()
        self.assertEqual(len(domain), 68)
        self.assertEqual(domain_env.observation_size, 68)

        strategic_env = StopContinueEnv(strategic_features=True, seed=7)
        strategic, _ = strategic_env.reset()
        self.assertEqual(len(strategic), 80)
        self.assertEqual(strategic_env.observation_size, 80)

    def test_potential_shaping_is_reported_separately(self):
        env = StopContinueEnv(seed=17, reward_shaping=0.25, shaping_gamma=0.99)
        env.reset()
        _, reward, terminated, _, info = env.step(StopContinueAction.STOP)

        self.assertFalse(terminated)
        self.assertEqual(info["extrinsic_reward"], 0.0)
        self.assertAlmostEqual(reward, info["shaping_reward"])
        self.assertNotEqual(reward, 0.0)

    def test_race_aware_potential_profile_is_available(self):
        env = StopContinueEnv(
            seed=17, reward_shaping=0.25, shaping_gamma=0.99,
            shaping_profile="race-aware",
        )
        env.reset()
        _, reward, terminated, _, info = env.step(StopContinueAction.STOP)

        self.assertFalse(terminated)
        self.assertEqual(info["extrinsic_reward"], 0.0)
        self.assertAlmostEqual(reward, info["shaping_reward"])


    def test_invalid_reward_shaping_configuration(self):
        with self.assertRaises(ValueError):
            StopContinueEnv(reward_shaping=-0.1)
        with self.assertRaises(ValueError):
            StopContinueEnv(shaping_gamma=1.1)
        with self.assertRaises(ValueError):
            StopContinueEnv(shaping_profile="unknown")

    def test_invalid_action_and_step_after_done_are_rejected(self):
        env = StopContinueEnv(seed=3)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(2)

        while env.game.winner is None:
            _, _, terminated, truncated, _ = env.step(StopContinueAction.STOP)
            if terminated or truncated:
                break
        with self.assertRaises(RuntimeError):
            env.step(StopContinueAction.STOP)

    def test_learner_can_use_second_seat(self):
        env = StopContinueEnv(learner_player=1, seed=81)

        observation, info = env.reset()

        self.assertEqual(len(observation), 58)
        self.assertEqual(env.game.current_player, 1)
        self.assertGreaterEqual(info["turns"][0], 1)

    def test_joint_mode_exposes_pairing_then_decision(self):
        env = StopContinueEnv(learn_pairing=True, seed=81)
        observation, _ = env.reset()
        self.assertEqual(len(observation), 76)
        self.assertEqual(env.action_size, 8)
        self.assertEqual(sum(env.legal_action_mask()), len(env.game.available))
        pairing = next(index for index, legal in enumerate(env.legal_action_mask()) if legal)
        observation, reward, terminated, truncated, _ = env.step(pairing)
        self.assertIsNotNone(observation)
        self.assertEqual(env.game.phase, "decision")
        self.assertEqual(env.legal_action_mask(), [False] * 6 + [True, True])
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        observation, _, _, _, _ = env.step(JointAction.STOP)
        self.assertTrue(observation is None or len(observation) == 76)

    def test_joint_pairing_action_receives_potential_shaping(self):
        env = StopContinueEnv(
            learn_pairing=True, reward_shaping=0.25, shaping_gamma=0.99, seed=81,
        )
        env.reset()
        previous_potential = env._potential()
        pairing = next(index for index, legal in enumerate(env.legal_action_mask()) if legal)

        _, reward, terminated, truncated, info = env.step(pairing)

        expected = 0.25 * (0.99 * env._potential() - previous_potential)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, expected)
        self.assertEqual(info["extrinsic_reward"], 0.0)
        self.assertAlmostEqual(info["shaping_reward"], expected)

    def test_joint_mode_reproduces_seeded_legal_actions(self):
        def trace(seed):
            env = StopContinueEnv(learn_pairing=True, seed=seed)
            observation, _ = env.reset()
            actions = []
            for _ in range(8):
                mask = env.legal_action_mask()
                action = next(index for index, legal in enumerate(mask) if legal)
                actions.append((tuple(observation), tuple(mask), action))
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            return actions
        self.assertEqual(trace(912), trace(912))


if __name__ == "__main__":
    unittest.main()
