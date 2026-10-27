import unittest

from cant_stop_core.paired_analysis import paired_bootstrap, paired_units


class TestPairedAnalysis(unittest.TestCase):
    def test_groups_alternating_games_by_seed(self):
        records = [
            {"seed": 2, "winner": "A"},
            {"seed": 2, "winner": "B"},
            {"seed": 3, "winner": "A"},
        ]
        self.assertEqual(paired_units(records), [(1, -1), (1,)])

    def test_bootstrap_is_seeded_and_handles_draws(self):
        records = [
            {"seed": 2, "winner": "A"},
            {"seed": 2, "winner": "B"},
            {"seed": 3, "winner": "A"},
            {"seed": 3, "winner": "draw"},
        ]
        first = paired_bootstrap(records, resamples=200, seed=9)
        second = paired_bootstrap(records, resamples=200, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(first["games"], 4)
        self.assertEqual(first["paired_units"], 2)
        self.assertEqual(first["a_wins"], 2)
        self.assertEqual(first["b_wins"], 1)
        self.assertEqual(first["draws"], 1)
        self.assertAlmostEqual(first["observed_margin_a_minus_b"], 0.25)


if __name__ == "__main__":
    unittest.main()
