import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestTrainingEntrypoint(unittest.TestCase):
    def test_short_reference_run_writes_checkpoint_and_metrics(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            result = subprocess.run([
                sys.executable, "scripts/train_dqn.py",
                "--backend", "python", "--algorithm", "double-dqn",
                "--pairing-policy", "ga", "--hidden-sizes", "16,16",
                "--episodes", "2", "--num-envs", "1", "--batch-size", "4",
                "--replay-capacity", "32", "--warmup-steps", "0",
                "--opponent-mix", "random:1", "--eval-every", "0",
                "--checkpoint-every", "0", "--validation-opponents", "random",
                "--interim-eval-games", "2", "--eval-games", "2",
                "--device", "cpu",
                "--torch-threads", "1", "--seed", "441",
                "--output-dir", str(output),
            ], cwd=root, capture_output=True, text=True, check=True)
            self.assertIn("Final-checkpoint", result.stdout)
            self.assertTrue((output / "final.pt").exists())
            self.assertTrue((output / "training_episodes.csv").exists())
            self.assertTrue((output / "evaluation_games.csv").exists())
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["backend"], "python-reference")

    def test_ppo_replaces_game_ending_before_second_seat_observation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ppo-smoke"
            result = subprocess.run([
                sys.executable, "scripts/train_ppo.py",
                "--pairing-policy", "learned",
                "--pairing-imitation-examples", "0",
                "--hidden-sizes", "16,16",
                "--episodes", "4", "--num-envs", "2",
                "--rollout-steps", "8", "--ppo-epochs", "1",
                "--minibatch-size", "4",
                "--opponent-mix", "mc:1",
                "--training-mc-rollouts", "16",
                "--mc-future-roll-limit", "20",
                "--validation-opponents", "random",
                "--validation-games", "2", "--eval-games", "2",
                "--eval-every", "100", "--progress-every", "100",
                # Seed 12103 (episode 1) lets MC win its opening turn before
                # the seat-1 learner acts, producing a valid empty episode.
                "--seed", "12102", "--output-dir", str(output),
            ], cwd=root, capture_output=True, text=True, check=True)
            self.assertIn("completed=4", result.stdout)
            self.assertTrue((output / "final.pt").exists())


if __name__ == "__main__":
    unittest.main()
