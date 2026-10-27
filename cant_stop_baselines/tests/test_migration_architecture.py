import unittest

from cant_stop_core.config import RuntimeConfig
from cant_stop_core.rl_env import StopContinueEnv
from cant_stop_core.transition import (
    TRANSITION_SCHEMA_VERSION,
    Transition,
    TransitionBatch,
)
from rollout.actor_pool import ActorPool


class TestTransitionSchema(unittest.TestCase):
    def test_terminal_next_state_is_fixed_size_zero_vector(self):
        transition = Transition.create(
            [1.0, 2.0], 1, 1.0, None, True,
            observation_size=2, episode_id=3, worker_id=1, step_id=4,
        )
        self.assertEqual(transition.next_state, (0.0, 0.0))
        self.assertEqual(transition.n_steps, 1)
        batch = TransitionBatch.from_iterable(2, [transition])
        self.assertEqual(batch.schema_version, TRANSITION_SCHEMA_VERSION)
        self.assertEqual(batch.replay_fields()[1], [1])

    def test_schema_rejects_wrong_feature_length(self):
        with self.assertRaises(ValueError):
            Transition.create([1.0], 0, 0.0, None, True, observation_size=2)

    def test_runtime_metadata_canonicalizes_python_alias(self):
        config = RuntimeConfig(backend="python", worker_count=1, base_seed=441)
        self.assertEqual(config.canonical_backend, "python-reference")
        self.assertEqual(config.metadata()["transition_schema_version"], 1)


class TestMultiprocessActors(unittest.TestCase):
    def test_basic_actor_protocol_matches_reference_trajectory(self):
        reference = StopContinueEnv(seed=772)
        observation, _ = reference.reset()
        with ActorPool(1) as pool:
            remote = pool.load_basic_many(((0, StopContinueEnv(seed=772)),))[0]
            self.assertEqual(remote, observation)
            for step in range(20):
                if observation is None:
                    break
                action = step % 2
                expected = reference.step(action)
                actual = pool.step_basic_many(((0, action),))[0]
                self.assertEqual(actual, expected[:4])
                observation = expected[0]

    def test_seeded_actor_matches_reference_trajectory(self):
        reference = StopContinueEnv(seed=991)
        observation, info = reference.reset()
        with ActorPool(1) as pool:
            remote = pool.load_many(((0, StopContinueEnv(seed=991)),))[0]
            self.assertEqual(remote.observation, observation)
            self.assertEqual(remote.info, info)
            for step in range(20):
                if observation is None:
                    break
                action = step % 2
                expected = reference.step(action)
                actual = pool.step_many(((0, action),))[0]
                self.assertEqual(actual.state.observation, expected[0])
                self.assertEqual(actual.reward, expected[1])
                self.assertEqual(actual.terminated, expected[2])
                self.assertEqual(actual.truncated, expected[3])
                self.assertEqual(actual.state.info, expected[4])
                observation = expected[0]


if __name__ == "__main__":
    unittest.main()
