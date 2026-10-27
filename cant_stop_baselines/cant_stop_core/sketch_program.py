"""Safe, dependency-free interpreter for saved Can't Stop sketch programs.

This is a project-local JSON AST format inspired by the public Medeiros DSL.
It is intentionally not presented as a byte-for-byte port of that repository's
Python AST.  The adapter is useful for executing persisted policies without
loading arbitrary Python or pickle data.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import COLUMNS, LENGTH
from .model import Move, State


class SketchProgramError(ValueError):
    """Raised for malformed or unsupported saved sketch programs."""


def _difficulty(columns: Sequence[int]) -> int:
    score = 0
    if all(column % 2 for column in columns):
        score += 7
    if all(column % 2 == 0 for column in columns):
        score += 1
    if all(column >= 7 for column in columns):
        score += 6
    if all(column <= 7 for column in columns):
        score += 5
    return score


def _environment(state: State, moves: Sequence[Move] = ()) -> dict[str, Any]:
    player = state.current_player
    permanent = state.permanent[player]
    active = state.active_dict()
    return {
        "state": state,
        "actions": list(moves),
        "active_columns": list(active),
        "completed": state.completed(player),
        "opponent_completed": state.completed(1 - player),
        "rolls_this_turn": state.rolls_this_turn,
        "progress": list(permanent),
        "column_heights": [LENGTH[column] for column in COLUMNS],
    }


def _value(node: Any, env: Mapping[str, Any]) -> Any:
    if isinstance(node, (int, float, bool, str)):
        return env[node] if isinstance(node, str) and node in env else node
    if not isinstance(node, Mapping):
        raise SketchProgramError(f"Expression must be a JSON value/object: {node!r}")
    if "const" in node:
        return node["const"]
    if "var" in node:
        name = node["var"]
        if name not in env:
            raise SketchProgramError(f"Unknown variable {name!r}")
        return env[name]
    op = node.get("op")
    if op in {"add", "sub", "mul"}:
        args = [_value(arg, env) for arg in node.get("args", [])]
        if len(args) < 2:
            raise SketchProgramError(f"{op} needs at least two arguments")
        if op == "add":
            return sum(args)
        if op == "sub":
            return args[0] - sum(args[1:])
        result = args[0]
        for arg in args[1:]:
            result *= arg
        return result
    if op in {"lt", "le", "gt", "ge", "eq"}:
        left = _value(node["left"], env)
        right = _value(node["right"], env)
        return {"lt": left < right, "le": left <= right,
                "gt": left > right, "ge": left >= right,
                "eq": left == right}[op]
    if op == "if":
        return _value(node["then"] if _value(node["condition"], env)
                      else node["else"], env)
    if op == "sum":
        return sum(_value(item, env) for item in node.get("items", []))
    if op == "argmax":
        items = env.get(node.get("over", "actions"))
        if not isinstance(items, Sequence) or not items:
            raise SketchProgramError("argmax requires a non-empty sequence")
        scores = []
        for item in items:
            child = dict(env)
            child["action"] = item
            scores.append(_value(node["score"], child))
        return max(range(len(scores)), key=lambda index: scores[index])
    if op == "action_columns":
        action = env.get("action")
        if not isinstance(action, Move):
            raise SketchProgramError("action_columns requires a Move action")
        return list(action.columns)
    if op == "action_advance":
        action = env.get("action")
        if not isinstance(action, Move):
            raise SketchProgramError("action_advance requires a Move action")
        column = int(node["column"])
        return action.columns.count(column)
    if op == "action_new_markers":
        action = env.get("action")
        if not isinstance(action, Move):
            raise SketchProgramError("action_new_markers requires a Move action")
        active = set(env["active_columns"])
        return len({column for column in action.columns if column not in active})
    if op == "difficulty":
        columns = node.get("columns", {"var": "active_columns"})
        return _difficulty(_value(columns, env))
    if op == "rule28_points":
        from .rules import rule28_points
        return rule28_points(env["state"])
    if op == "bust_probability":
        from .rules import bust_probability
        return bust_probability(env["state"])
    raise SketchProgramError(f"Unsupported expression operation {op!r}")


class SketchProgramAgent:
    """Execute a persisted JSON sketch using the existing Agent protocol.

    ``pair`` must evaluate to an action index, normally using ``argmax``.
    ``stop`` must evaluate to a truthy value.  Pair selection and stop policy
    remain separate, matching the current engine's two callbacks.
    """

    def __init__(self, program: Mapping[str, Any]):
        if program.get("version") != 1:
            raise SketchProgramError("Only sketch program version 1 is supported")
        if not isinstance(program.get("pair"), Mapping) \
                or not isinstance(program.get("stop"), Mapping):
            raise SketchProgramError("Program requires object-valued pair and stop roots")
        self.program = dict(program)

    @classmethod
    def from_json(cls, path: str | Path) -> "SketchProgramAgent":
        with Path(path).open(encoding="utf-8") as handle:
            program = json.load(handle)
        if not isinstance(program, Mapping):
            raise SketchProgramError("Top-level JSON program must be an object")
        return cls(program)

    def choose_move(self, state: State, moves: Sequence[Move],
                    rng: random.Random) -> Move:
        if not moves:
            raise SketchProgramError("Cannot choose from an empty move list")
        index = _value(self.program["pair"], _environment(state, moves))
        if not isinstance(index, int) or isinstance(index, bool) \
                or not 0 <= index < len(moves):
            raise SketchProgramError(f"Pair program returned invalid index {index!r}")
        return moves[index]

    def should_stop(self, state: State, rng: random.Random) -> bool:
        return bool(_value(self.program["stop"], _environment(state)))
