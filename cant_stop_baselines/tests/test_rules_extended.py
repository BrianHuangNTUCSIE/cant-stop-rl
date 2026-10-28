import itertools
import random
import unittest

from cant_stop import COLUMNS, LENGTH, Game, Move, State, bust_probability, legal_moves
from cant_stop.rules import _legal_move_columns_fast


def reference_moves(dice, permanent, owner, active):
    """Small sequential oracle independent of the production move generator."""

    def step(column, positions):
        if owner[column - 2] != -1:
            return None
        if column not in positions and len(positions) >= 3:
            return None
        position = positions.get(column, permanent[column - 2])
        if position >= LENGTH[column]:
            return None
        result = positions.copy()
        result[column] = position + 1
        return result

    a, b, c, d = dice
    partitions = ((a + b, c + d), (a + c, b + d), (a + d, b + c))
    outcomes = []

    def add(outcome):
        if outcome not in outcomes:
            outcomes.append(outcome)
    for left, right in partitions:
        after_left = step(left, active)
        after_right = step(right, active)
        both_possible = False
        if after_left is not None and step(right, after_left) is not None:
            both_possible = True
        if after_right is not None and step(left, after_right) is not None:
            both_possible = True
        if both_possible:
            add(Move(tuple(sorted((left, right)))))
        else:
            if after_left is not None:
                add(Move((left,)))
            if after_right is not None:
                add(Move((right,)))
    return tuple(outcomes)


class TestMoveGenerationExhaustive(unittest.TestCase):
    def test_fast_rollout_legal_moves_match_public_path(self):
        rng = random.Random(441)
        for _ in range(500):
            permanent = [rng.randrange(LENGTH[column] + 1) for column in COLUMNS]
            owner = [rng.choice((-1, 0, 1)) for _ in COLUMNS]
            active = {}
            for column in COLUMNS:
                if (rng.random() < 0.25
                        and permanent[column - 2] < LENGTH[column]):
                    active[column] = rng.randint(
                        permanent[column - 2] + 1, LENGTH[column]
                    )
            dice = tuple(rng.randint(1, 6) for _ in range(4))
            expected = tuple(
                move.columns for move in legal_moves(dice, permanent, owner, active)
            )
            actual = _legal_move_columns_fast(dice, permanent, owner, active)
            self.assertEqual(actual, expected)

    def test_all_dice_match_sequential_oracle_in_representative_states(self):
        blank = (0,) * 11
        open_columns = (-1,) * 11

        near_tops = list(blank)
        near_tops[5 - 2] = LENGTH[5] - 1

        owner = list(open_columns)
        owner[7 - 2] = 1

        states = (
            (blank, open_columns, {}),
            (blank, open_columns, {5: 1, 6: 1}),
            (blank, open_columns, {5: 1, 7: 1, 9: 1}),
            (tuple(near_tops), open_columns, {5: LENGTH[5]}),
            (blank, tuple(owner), {4: LENGTH[4] - 1, 8: 2}),
        )

        for permanent, column_owner, active in states:
            for dice in itertools.product(range(1, 7), repeat=4):
                with self.subTest(state=states.index((permanent, column_owner, active)),
                                  dice=dice):
                    self.assertEqual(
                        legal_moves(dice, permanent, column_owner, active),
                        reference_moves(dice, permanent, column_owner, active),
                    )

    def test_generated_moves_preserve_marker_and_height_limits(self):
        permanent = (0,) * 11
        owner = (-1,) * 11
        active = {4: LENGTH[4] - 1, 7: 3}
        original = active.copy()

        for dice in itertools.product(range(1, 7), repeat=4):
            for move in legal_moves(dice, permanent, owner, active):
                result = active.copy()
                for column in move.columns:
                    result[column] = result.get(column, permanent[column - 2]) + 1
                self.assertLessEqual(len(result), 3)
                self.assertTrue(all(position <= LENGTH[column]
                                    for column, position in result.items()))
        self.assertEqual(active, original, "legal_moves must not mutate caller state")

    def test_fast_bust_probability_matches_exhaustive_legal_moves(self):
        blank = (0,) * 11
        open_columns = (-1,) * 11
        near_tops = list(blank)
        near_tops[5 - 2] = LENGTH[5] - 1
        owner = list(open_columns)
        owner[7 - 2] = 1
        states = (
            (blank, open_columns, {}),
            (blank, open_columns, {5: 1, 6: 1}),
            (blank, open_columns, {5: 1, 7: 1, 9: 1}),
            (tuple(near_tops), open_columns, {5: LENGTH[5]}),
            (blank, tuple(owner), {4: LENGTH[4] - 1, 8: 2}),
        )

        for permanent, column_owner, active in states:
            expected_busts = sum(
                not legal_moves(dice, permanent, column_owner, active)
                for dice in itertools.product(range(1, 7), repeat=4)
            )
            state = State(
                permanent=(permanent, blank),
                owner=column_owner,
                active=tuple(sorted(active.items())),
                current_player=0,
                rolls_this_turn=1,
            )
            self.assertEqual(bust_probability(state), expected_busts / 1296.0)


class TestGameInvariants(unittest.TestCase):
    def test_snapshot_is_not_changed_by_later_game_mutation(self):
        game = Game()
        before = game.snapshot()
        game.roll(random.Random(1), (1, 2, 3, 4))
        game.apply_move(Move((5, 5)))

        self.assertEqual(before.active, ())
        self.assertEqual(before.permanent, ((0,) * 11, (0,) * 11))

    def test_stop_can_claim_multiple_columns_and_third_claim_wins(self):
        game = Game()
        game.owner[2 - 2] = 0
        game.permanent[0][3 - 2] = LENGTH[3] - 1
        game.permanent[0][4 - 2] = LENGTH[4] - 1
        game.active = {3: LENGTH[3], 4: LENGTH[4]}
        game.phase = "decision"

        game.stop()

        self.assertEqual(game.owner[3 - 2], 0)
        self.assertEqual(game.owner[4 - 2], 0)
        self.assertEqual(game.winner, 0)
        self.assertEqual(game.phase, "done")

    def test_column_owned_by_opponent_is_closed_to_current_player(self):
        permanent = (0,) * 11
        owner = [-1] * 11
        owner[6 - 2] = 1

        moves = legal_moves((3, 3, 3, 3), permanent, owner, {})

        self.assertEqual(moves, ())


if __name__ == "__main__":
    unittest.main()
