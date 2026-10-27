import argparse
import unittest
from pathlib import Path

from evaluate_ml_head_to_head import parse_baseline_spec, parse_model_spec


class TestHeadToHeadModelSpecs(unittest.TestCase):
    def test_defaults_name_to_file_stem(self):
        spec = parse_model_spec("runs/example/best.pt", "dqn")
        self.assertEqual(spec.name, "best")
        self.assertEqual(spec.kind, "dqn")
        self.assertEqual(spec.path, Path("runs/example/best.pt"))

    def test_explicit_name_allows_repeated_checkpoint_filenames(self):
        spec = parse_model_spec("ddqn-seed441=runs/seed441/best.pt", "dqn")
        self.assertEqual(spec.name, "ddqn-seed441")
        self.assertEqual(spec.path, Path("runs/seed441/best.pt"))

    def test_rejects_name_that_is_unsafe_for_output_filename(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_model_spec("seed/441=runs/seed441/best.pt", "dqn")

    def test_builtin_agent_has_no_model_path(self):
        spec = parse_baseline_spec("rule28-smart")
        self.assertEqual(spec.name, "rule28-smart")
        self.assertEqual(spec.kind, "baseline")
        self.assertIsNone(spec.path)


if __name__ == "__main__":
    unittest.main()
