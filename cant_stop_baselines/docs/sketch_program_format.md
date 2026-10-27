# Project Sketch JSON format

This is a small, safe inference format for `SketchProgramAgent`. It is not a
serialized copy of the Medeiros Python AST and does not execute Python or
unpickle demonstrations.

Every program has `version: 1`, a `pair` expression returning an index into the
current legal `actions` list, and a truth-valued `stop` expression. Supported
operations include `argmax`, `add`, `sub`, `mul`, comparisons, `if`, `sum`,
`action_advance`, `action_new_markers`, `difficulty`, `rule28_points`, and
`bust_probability`.

Example:

```json
{
  "version": 1,
  "pair": {
    "op": "argmax",
    "over": "actions",
    "score": {"op": "action_advance", "column": 7}
  },
  "stop": {
    "op": "ge",
    "left": {"op": "rule28_points"},
    "right": 28
  }
}
```

Use it with `--sketch-program-a` or `--sketch-program-b`. A synthesized or
imported program must retain its source revision, demonstrations, seed,
search budget, and AST text in surrounding experiment metadata; this adapter
does not make such provenance claims automatically.
