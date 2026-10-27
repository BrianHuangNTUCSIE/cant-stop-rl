# Can't Stop: Reusable Game Engine and Research Agents

This repository contains a deterministic two-player *Can't Stop* engine,
baseline and learned agents, evaluation utilities, and the experiment code
used by the accompanying research project.

## Repository guide

- `cant_stop_baselines/cant_stop_core/` — reusable rules, state model, engine,
  agents, RL environment, and evaluation API.
- `cant_stop_baselines/scripts/` — training, evaluation, dataset, and analysis
  entry points.
- `cant_stop_baselines/tests/` — rule, transition, reproducibility, agent, and
  training tests.
- `cant_stop_baselines/docs/` — experiment status and technical audits.
- `final_report/` — paper source and generated figures/PDF.
- `spec.pdf` — course specification; it is not part of the engine contract.

The compatibility-first migration strategy and target architecture are in
[`docs/REFACTOR_GUIDE.md`](docs/REFACTOR_GUIDE.md).

## Quick start

The current source tree remains runnable directly:

```bash
cd cant_stop_baselines
python3 -m unittest discover -s tests -t . -v
python3 cant_stop.py --agent-a rule28-smart --agent-b random --games 100 --seed 42
```

For an editable install from the repository root:

```bash
python3 -m pip install -e .
cant-stop --agent-a rule28-smart --agent-b random --games 100 --seed 42
```

Optional ML and RL dependencies are documented in
`cant_stop_baselines/requirements-ml.txt` and
`cant_stop_baselines/requirements-rl.txt`, and are also available as the
`.[ml]` and `.[rl]` installation extras.

## Reproducibility

Experiments must save their configuration, code version, seeds, raw game-level
CSV records, and machine/dependency metadata. CSV files are retained when they
support plots or statistical analysis. Runtime logs, caches, virtual
environments, checkpoints, and generated report intermediates are local
artifacts and are ignored by Git.

Read `cant_stop_baselines/docs/EXPERIMENT_STATUS.md` before interpreting or
extending the existing results. The current evaluator alternates seats and
uses paired dice seeds; use paired analysis or independent-seed variability,
not naive independent-game confidence intervals.

## Citation and reuse

Before publication, add the final paper citation, license, and release DOI.
Changes to game rules, feature layouts, random-seed semantics, or evaluation
protocol should be versioned and documented as compatibility changes.
