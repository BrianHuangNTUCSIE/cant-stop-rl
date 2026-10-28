import random
import unittest

from cant_stop import (
    COLUMNS, LENGTH, Move, Game, State, CoutoAgent, GlennAloiAgent,
    Glenn2009ConstantAgent, Glenn2010LinearAgent,
    RandomAgent, Rule28Agent,
    MonteCarloAgent, FullGameMonteCarloAgent, StrongMonteCarloAgent,
    VMerckleMonteAgent, VMerckleMonteUCTAgent,
    SketchProgramAgent, SketchProgramError, carter_features,
    legal_moves, rule28_points, bust_probability,
    pair_score, play_game, evaluate, make_agent,
)
from cant_stop.carter_adapter import CarterKerasAgent


class TestCantStop(unittest.TestCase):
    def setUp(self):
        self.permanent = (0,) * 11
        self.owner = (-1,) * 11

    def test_pairings(self):
        moves = legal_moves((1, 2, 3, 4), self.permanent, self.owner, {})
        self.assertEqual(set(moves), {Move((3, 7)), Move((4, 6)), Move((5, 5))})

    def test_pair_dedup(self):
        self.assertEqual(legal_moves((4, 4, 4, 4), self.permanent, self.owner, {}),
                         (Move((8, 8)),))

    def test_one_slot_two_new_numbers(self):
        active = {5: 1, 6: 1}
        moves = legal_moves((1, 2, 3, 4), self.permanent, self.owner, active)
        self.assertIn(Move((3,)), moves)  # 3 + 7, only one runner free
        self.assertIn(Move((7,)), moves)
        self.assertIn(Move((4, 6)), moves)  # 6 is already active
        self.assertIn(Move((5, 5)), moves)
        self.assertNotIn(Move((3, 7)), moves)

    def test_both_mandatory_if_possible(self):
        moves = legal_moves((1, 2, 3, 4), self.permanent, self.owner, {})
        self.assertNotIn(Move((3,)), moves)
        self.assertNotIn(Move((7,)), moves)

    def test_only_one_valid_sum(self):
        active = {5: 1, 6: 1, 8: 1}
        moves = legal_moves((1, 2, 3, 4), self.permanent, self.owner, active)
        self.assertEqual(set(moves), {Move((6,)), Move((5, 5))})

    def test_duplicate_when_one_space_left(self):
        active = {5: LENGTH[5] - 1}
        # (2+3,2+3) allows only one step on 5; (2+2,3+3) remains legal.
        moves = legal_moves((2, 2, 3, 3), self.permanent, self.owner, active)
        self.assertIn(Move((5,)), moves)
        self.assertNotIn(Move((5, 5)), moves)

    def test_closed_column_cannot_move(self):
        owner = list(self.owner)
        owner[5-2] = 0
        self.assertNotIn(Move((5, 5)), legal_moves((2, 2, 3, 3),
                                                   self.permanent, owner, {}))

    def test_stop_banks_and_switches(self):
        game = Game()
        choices = game.roll(random.Random(2), (1, 2, 3, 4))
        self.assertIn(Move((5, 5)), choices)
        game.apply_move(Move((5, 5)))
        self.assertEqual(game.active, {5: 2})
        self.assertEqual(game.permanent[0][3], 0)
        game.stop()
        self.assertEqual(game.permanent[0][3], 2)
        self.assertEqual(game.current_player, 1)
        self.assertEqual(game.active, {})

    def test_bust_discards_temporary_progress(self):
        game = Game()
        game.active = {5: 3, 7: 2, 9: 1}
        game.permanent[0][3] = 1
        choices = game.roll(random.Random(1), (1, 1, 1, 1))
        self.assertFalse(choices)
        self.assertEqual(game.permanent[0][3], 1)
        self.assertEqual(game.busts[0], 1)
        self.assertEqual(game.current_player, 1)
        self.assertEqual(game.active, {})

    def test_closes_only_after_bank(self):
        game = Game()
        game.permanent[0][0] = 2
        choices = game.roll(random.Random(1), (1, 1, 1, 1))
        self.assertIn(Move((2,)), choices)
        game.apply_move(Move((2,)))
        self.assertEqual(game.active[2], 3)
        self.assertEqual(game.owner[0], -1)
        game.stop()
        self.assertEqual(game.owner[0], 0)

    def test_banked_third_claim_wins(self):
        game = Game()
        game.owner[0] = 0
        game.owner[1] = 0
        game.permanent[0][2] = LENGTH[4] - 1
        choices = game.roll(random.Random(1), (1, 1, 2, 2))
        self.assertIn(Move((4,)), choices)
        game.apply_move(Move((4,)))
        self.assertIsNone(game.winner)
        game.stop()
        self.assertEqual(game.winner, 0)
        self.assertEqual(game.phase, 'done')

    def test_rule_of_28_example(self):
        # 5 marked+advanced (9) + 3 marked (10) + 6 marked (4)
        # + all below 8 (4) = 27. Then advance 6 gives 29.
        game = Game()
        game.active = {3: 1, 5: 2, 6: 1}
        self.assertEqual(rule28_points(game.snapshot()), 27)
        game.active[6] = 2
        self.assertEqual(rule28_points(game.snapshot()), 29)

    def test_correct_ga_7_weight(self):
        game = Game()
        self.assertEqual(pair_score(game.snapshot(), Move((7, 7)), 'ga'),
                         3+3-6)

    def test_features_dimensions(self):
        game = Game()
        self.assertEqual(len(game.snapshot().features()), 58)
        self.assertEqual(len(game.snapshot().features(enhanced=True)), 60)
        self.assertEqual(len(game.snapshot().features(domain=True)), 68)
        self.assertEqual(len(game.snapshot().features(strategic=True)), 80)

    def test_bust_probability_known_columns(self):
        game = Game()
        game.active = {6: 1, 7: 1, 8: 1}
        p = bust_probability(game.snapshot())
        self.assertGreater(p, 0)
        self.assertAlmostEqual(p, 0.0802, delta=0.01)

    def test_match_reproducibility(self):
        agents = (Rule28Agent(), RandomAgent())
        self.assertEqual(play_game(agents, 123), play_game(agents, 123))

    def test_published_strong_agent_pairing_policies(self):
        permanent = [list(self.permanent), list(self.permanent)]
        permanent[0][5-2] = 8
        state = State(tuple(tuple(row) for row in permanent), self.owner, (), 0, 0)
        moves = (Move((5,)), Move((6,)))
        rng = random.Random(1)
        self.assertEqual(CoutoAgent().choose_move(state, moves, rng), Move((5,)))
        self.assertEqual(CoutoAgent(pair_heuristic="ga").choose_move(state, moves, rng),
                         Move((6,)))
        self.assertEqual(GlennAloiAgent().choose_move(state, moves, rng), Move((6,)))

    def test_published_strong_agent_stop_formulas_differ(self):
        state = State((self.permanent, self.permanent), self.owner,
                      ((2, 1), (7, 1), (12, 1)), 0, 1)
        rng = random.Random(1)
        self.assertTrue(GlennAloiAgent().should_stop(state, rng))
        self.assertFalse(CoutoAgent().should_stop(state, rng))

        advanced = State((self.permanent, self.permanent), self.owner,
                         ((2, 2), (7, 6), (12, 2)), 0, 2)
        self.assertTrue(CoutoAgent().should_stop(advanced, rng))

    def test_published_agents_retain_top_neutral_until_bank(self):
        state = State((self.permanent, self.permanent), self.owner,
                      ((2, LENGTH[2]), (3, 1), (4, 1)), 0, 1)
        self.assertTrue(GlennAloiAgent().should_stop(state, random.Random(1)))
        self.assertTrue(CoutoAgent().should_stop(state, random.Random(1)))

    def test_carter_uses_source_partition_order(self):
        state = State((self.permanent, self.permanent), self.owner,
                      ((2, 1), (3, 1)), 0, 1)
        moves = legal_moves((1, 2, 1, 2), self.permanent, self.owner,
                            state.active_dict())
        self.assertEqual(moves, (Move((3, 3)), Move((2, 4))))
        agent = object.__new__(CarterKerasAgent)
        self.assertEqual(agent.choose_move(state, moves, random.Random(1)),
                         Move((3, 3)))

    def test_published_agents_bank_a_winning_turn(self):
        owner = list(self.owner)
        owner[0] = owner[1] = 0
        state = State((self.permanent, self.permanent), tuple(owner),
                      ((4, LENGTH[4]),), 0, 1)
        self.assertTrue(GlennAloiAgent().should_stop(state, random.Random(1)))
        self.assertTrue(CoutoAgent().should_stop(state, random.Random(1)))

    def test_glenn_2009_constant_paper_scores(self):
        state = State((self.permanent, self.permanent), self.owner,
                      ((6, 1), (8, 1), (10, 1)), 0, 1)
        agent = Glenn2009ConstantAgent()
        # Progress is 2*(2+2+3)=14; the all-even bonus adds 1.
        self.assertEqual(agent._stop_score(state), 15)
        self.assertFalse(agent.should_stop(state, random.Random(1)))
        self.assertEqual(
            agent._move_score(
                State((self.permanent, self.permanent), self.owner, (), 0, 0),
                Move((7, 7)),
            ),
            0,  # 3+3 minus one new-marker penalty of 6
        )

    def test_glenn_2010_linear_table_6_scores(self):
        state = State((self.permanent, self.permanent), self.owner,
                      ((3, 1), (5, 1), (7, 1)), 0, 1)
        agent = Glenn2010LinearAgent()
        # Progress is 2*(13+8+3)=48; all-odd and all-low add 11+4.
        self.assertEqual(agent._stop_score(state), 63)
        self.assertTrue(agent.should_stop(state, random.Random(1)))
        empty = State((self.permanent, self.permanent), self.owner, (), 0, 0)
        # Column 2, first space: floor(72*1/3 + 15)=39, less marker penalty 12.
        self.assertEqual(agent._move_score(empty, Move((2,))), 27)

    def test_evaluator_consistency(self):
        result = evaluate(Rule28Agent(), RandomAgent(), games=24, seed=99)
        self.assertEqual(result['agents'][0]['wins'] + result['agents'][1]['wins']
                         + result['draws'], 24)
        self.assertEqual(sum(x['total_turns'] for x in result['agents']),
                         sum(x['total_busts'] + x['total_stops'] for x in result['agents']))

    def test_monte_carlo_completes_game(self):
        result = play_game((MonteCarloAgent(rollouts=3, future_roll_limit=8),
                            RandomAgent()), seed=119)
        self.assertIn(result.winner, (0, 1))

    def test_full_game_monte_carlo_completes_game(self):
        result = play_game((FullGameMonteCarloAgent(rollouts=2, max_turns=100),
                            RandomAgent()), seed=119, max_total_turns=100)
        self.assertIn(result.winner, (0, 1, None))

    def test_full_game_monte_carlo_is_named_separately(self):
        self.assertIsInstance(make_agent("mc-full-game", 2),
                              FullGameMonteCarloAgent)
        self.assertIsInstance(make_agent("mc-strong", 2), StrongMonteCarloAgent)

    def test_vmerckle_search_agents_are_distinct_and_seeded(self):
        for agent_type, name in ((VMerckleMonteAgent, "vmerckle_monte"),
                                 (VMerckleMonteUCTAgent, "vmerckle_monte_uct")):
            self.assertIsInstance(make_agent(name, 4), agent_type)
            result = play_game((agent_type(rollouts=4), RandomAgent()),
                               seed=211, max_total_turns=100)
            repeat = play_game((agent_type(rollouts=4), RandomAgent()),
                               seed=211, max_total_turns=100)
            self.assertEqual(result, repeat)

    def test_vmerckle_search_rejects_invalid_budget(self):
        with self.assertRaises(ValueError):
            play_game((VMerckleMonteAgent(rollouts=0), RandomAgent()), seed=1)

    def test_sketch_json_adapter_executes_pair_and_stop_roots(self):
        program = {
            "version": 1,
            "pair": {
                "op": "argmax", "over": "actions",
                "score": {"op": "action_advance", "column": 7},
            },
            "stop": {"op": "ge", "left": {"op": "rule28_points"},
                     "right": 28},
        }
        agent = SketchProgramAgent(program)
        state = State((self.permanent, self.permanent), self.owner,
                      ((5, 1),), 0, 1)
        moves = (Move((5,)), Move((7,)))
        self.assertEqual(agent.choose_move(state, moves, random.Random(1)),
                         Move((7,)))
        self.assertFalse(agent.should_stop(state, random.Random(1)))

    def test_sketch_json_adapter_rejects_invalid_program(self):
        with self.assertRaises(SketchProgramError):
            SketchProgramAgent({"version": 2, "pair": {}, "stop": {}})

    def test_carter_feature_adapter_has_verified_width_and_scaling(self):
        state = State((self.permanent, self.permanent), self.owner,
                      ((5, 1), (7, 2)), 0, 2)
        features = carter_features(state)
        self.assertEqual(len(features), 51)
        self.assertEqual(features[2 * 11 + (5 - 2)], 1 / LENGTH[5])
        self.assertEqual(features[3 * 11], 0.0)
        self.assertEqual(features[-3:], [0.0, 0.0, 0.0])

    def test_bad_actions_rejected(self):
        game = Game()
        with self.assertRaises(RuntimeError):
            game.stop()
        game.roll(random.Random(2), (1, 2, 3, 4))
        with self.assertRaises(ValueError):
            game.apply_move(Move((2,)))


if __name__ == '__main__':
    unittest.main()
