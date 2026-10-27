import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from cant_stop.game import Game
from cant_stop.rules import legal_moves


class TestNativeRules(unittest.TestCase):
    """Differential smoke tests for the optional first native milestone."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("clang++") is None and shutil.which("g++") is None:
            raise unittest.SkipTest("no C++ compiler available")
        root = Path(__file__).resolve().parents[1]
        compiler = shutil.which("clang++") or shutil.which("g++")
        cls._tempdir = tempfile.TemporaryDirectory()
        cls._binary = Path(cls._tempdir.name) / "rules_cli"
        subprocess.run([
            compiler, "-std=c++17", "-O2", "-I", str(root / "native"),
            str(root / "native" / "rules_cli.cpp"), "-o", str(cls._binary),
        ], check=True, capture_output=True, text=True)
        cls._transition_binary = Path(cls._tempdir.name) / "transition_cli"
        subprocess.run([
            compiler, "-std=c++17", "-O2", "-I", str(root / "native"),
            str(root / "native" / "transition_cli.cpp"),
            "-o", str(cls._transition_binary),
        ], check=True, capture_output=True, text=True)
        cls._trajectory_binary = Path(cls._tempdir.name) / "trajectory_cli"
        subprocess.run([
            compiler, "-std=c++17", "-O2", "-I", str(root / "native"),
            str(root / "native" / "trajectory_cli.cpp"),
            "-o", str(cls._trajectory_binary),
        ], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_tempdir"):
            cls._tempdir.cleanup()

    def test_random_states_match_python_legal_moves(self):
        rng = random.Random(441)
        cases = []
        expected = []
        for _ in range(200):
            game = Game()
            game.permanent = [[rng.randrange(0, 3) for _ in range(11)] for _ in range(2)]
            game.owner = [rng.choice((-1, 0, 1)) for _ in range(11)]
            game.active = {}
            for column in rng.sample(range(2, 13), rng.randrange(4)):
                if len(game.active) == 3:
                    break
                game.active[column] = min(game.permanent[0][column - 2] + 1, {
                    2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13,
                    8: 11, 9: 9, 10: 7, 11: 5, 12: 3,
                }[column])
            dice = tuple(rng.randint(1, 6) for _ in range(4))
            state = game.snapshot()
            cases.append((*dice, *state.permanent[0], *state.owner,
                          len(state.active), *sum(([c, p] for c, p in state.active), [])))
            expected.append(tuple(move.columns for move in legal_moves(
                dice, state.permanent[0], state.owner, state.active_dict())))

        payload = "\n".join(" ".join(map(str, case)) for case in cases) + "\n"
        output = subprocess.run(
            [str(self._binary)], input=payload, text=True,
            capture_output=True, check=True,
        ).stdout.splitlines()
        actual = []
        for line in output:
            values = list(map(int, line.split()))
            count = values[0]
            cursor = 1
            moves = []
            for _ in range(count):
                size = values[cursor]
                cursor += 1
                moves.append(tuple(values[cursor:cursor + size]))
                cursor += size
            actual.append(tuple(moves))
        self.assertEqual(actual, expected)

    def test_advance_active_matches_python(self):
        rng = random.Random(442)
        cases = []
        expected = []
        lengths = (3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3)
        for _ in range(200):
            game = Game()
            game.permanent = [[rng.randrange(0, 3) for _ in range(11)] for _ in range(2)]
            game.owner = [-1] * 11
            game.active = {}
            for column in rng.sample(range(2, 13), rng.randrange(4)):
                if len(game.active) == 3:
                    break
                game.active[column] = min(
                    game.permanent[0][column - 2] + 1, lengths[column - 2]
                )
            state = game.snapshot()
            dice = tuple(rng.randint(1, 6) for _ in range(4))
            moves = legal_moves(
                dice, state.permanent[0], state.owner, state.active_dict()
            )
            if not moves:
                continue
            move = rng.choice(moves)
            cases.append((*dice, *state.permanent[0], *state.owner,
                          len(state.active), *sum(([c, p] for c, p in state.active), []),
                          len(move.columns), *move.columns))
            next_active = dict(state.active_dict())
            for column in move.columns:
                next_active[column] = next_active.get(
                    column, state.permanent[0][column - 2]
                ) + 1
            expected.append(tuple(sorted(next_active.items())))

        payload = "\n".join(" ".join(map(str, case)) for case in cases) + "\n"
        output = subprocess.run(
            [str(self._transition_binary)], input=payload, text=True,
            capture_output=True, check=True,
        ).stdout.splitlines()
        actual = []
        for line in output:
            values = list(map(int, line.split()))
            count = values[0]
            actual.append(tuple((values[index], values[index + 1])
                                for index in range(1, 1 + 2 * count, 2)))
        self.assertEqual(actual, expected)

    def test_explicit_game_trajectories_match_python(self):
        rng = random.Random(443)
        payload_cases = []
        expected = []
        for _ in range(40):
            game = Game()
            commands = []
            for _ in range(80):
                dice = tuple(rng.randint(1, 6) for _ in range(4))
                choices = game.roll(random.Random(0), dice)
                if not choices:
                    commands.append((*dice, 0, 0))
                    if game.winner is not None:
                        break
                    continue
                move = choices[0]
                decision = rng.randrange(2)
                commands.append((*dice, len(move.columns), *move.columns, decision))
                game.apply_move(move)
                if decision:
                    game.stop()
                else:
                    game.continue_turn()
                if game.winner is not None:
                    break
            payload_cases.append(commands)
            active = tuple(sorted(game.active.items()))
            expected.append((
                {"roll": 0, "pair": 1, "decision": 2, "done": 3}[game.phase],
                game.current_player, -1 if game.winner is None else game.winner,
                game.rolls_this_turn, *game.permanent[0], *game.permanent[1],
                *game.owner, len(active), *sum(([c, p] for c, p in active), []),
                *game.turns, *game.rolls, *game.busts, *game.stops,
            ))

        payload = "".join(
            str(len(commands)) + "\n" +
            "\n".join(" ".join(map(str, command)) for command in commands) + "\n"
            for commands in payload_cases
        )
        output = subprocess.run(
            [str(self._trajectory_binary)], input=payload, text=True,
            capture_output=True, check=True,
        ).stdout.splitlines()
        actual = [tuple(map(int, line.split())) for line in output]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
