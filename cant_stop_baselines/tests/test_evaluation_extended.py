import random
import unittest

from cant_stop import Move, RandomAgent, Rule28Agent, evaluate, play_game


class AlwaysContinueAgent:
    def choose_move(self, state, moves, rng: random.Random) -> Move:
        return moves[0]

    def should_stop(self, state, rng: random.Random) -> bool:
        return False


class TestEvaluationProtocol(unittest.TestCase):
    def test_game_limit_finishes_one_turn_then_records_draw(self):
        result = play_game(
            (AlwaysContinueAgent(), AlwaysContinueAgent()),
            seed=17,
            max_total_turns=1,
        )

        self.assertIsNone(result.winner)
        self.assertEqual(result.turns, (1, 0))
        self.assertEqual(result.busts, (1, 0))

    def test_records_use_paired_seeds_and_alternate_seats(self):
        report = evaluate(
            Rule28Agent(),
            RandomAgent(),
            games=5,
            seed=100,
            include_game_records=True,
        )
        records = report["game_records"]

        self.assertEqual([row["seed"] for row in records],
                         [100, 100, 101, 101, 102])
        self.assertEqual([row["a_seat"] for row in records], [0, 1, 0, 1, 0])

    def test_raw_records_aggregate_to_summary(self):
        report = evaluate(
            Rule28Agent(),
            RandomAgent(),
            games=12,
            seed=441,
            include_game_records=True,
        )
        records = report["game_records"]

        for index, label in enumerate(("a", "b")):
            summary = report["agents"][index]
            self.assertEqual(summary["wins"],
                             sum(row["winner"] == label.upper() for row in records))
            self.assertEqual(summary["total_turns"],
                             sum(row[f"{label}_turns"] for row in records))
            self.assertEqual(summary["total_rolls"],
                             sum(row[f"{label}_rolls"] for row in records))
            self.assertEqual(summary["total_busts"],
                             sum(row[f"{label}_busts"] for row in records))
            self.assertEqual(summary["total_stops"],
                             sum(row[f"{label}_stops"] for row in records))
            self.assertEqual(summary["wins"],
                             summary["seat0_wins"] + summary["seat1_wins"])

    def test_progress_callback_runs_after_each_game(self):
        updates = []
        evaluate(
            Rule28Agent(), RandomAgent(), games=4, seed=700,
            progress_callback=lambda completed, total: updates.append(
                (completed, total)
            ),
        )
        self.assertEqual(updates, [(1, 4), (2, 4), (3, 4), (4, 4)])


if __name__ == "__main__":
    unittest.main()
