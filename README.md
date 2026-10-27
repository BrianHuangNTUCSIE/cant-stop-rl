# Can't Stop: Reusable Game Engine and Research Agents

This repository contains a deterministic two-player *Can't Stop* engine,
baseline and learned agents, evaluation utilities, and the experiment code
used by the accompanying research project.

## Repository guide

- `src/cant_stop/` — reusable rules, state model, engine,
  agents, RL environment, and evaluation API.
- `cant_stop_baselines/scripts/` — training, evaluation, dataset, and analysis
  entry points.
- `cant_stop_baselines/tests/` — rule, transition, reproducibility, agent, and
  training tests.
- `cant_stop_baselines/docs/` — experiment status and technical audits.
- `final_report/` — paper source and generated figures/PDF.
- `spec.pdf` — course specification; it is not part of the engine contract.

The clean-checkout verification and experiment recording procedure is in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Quick start

The source tree is installed as a normal `src/` package:

```bash
cd cant_stop_baselines
python3 -m unittest discover -s tests -t . -v
cant-stop --agent-a rule28-smart --agent-b random --games 100 --seed 42
```

For an editable install from the repository root:

```bash
python3 -m pip install -e .
cant-stop --agent-a rule28-smart --agent-b random --games 100 --seed 42
```

The same checks are available through the root developer commands:

```bash
make verify
```

Use `make install-ml` or `make install-rl` when running the corresponding
optional experiment families.

Optional ML and RL dependencies are documented in
`cant_stop_baselines/requirements-ml.txt` and
`cant_stop_baselines/requirements-rl.txt`, and are also available as the
`.[ml]` and `.[rl]` installation extras.

## Reproducibility

Experiments must save their configuration, code version, seeds, raw game-level
CSV records, and machine/dependency metadata. Model checkpoints belong under
`runs/`; evaluation summaries and raw game records belong under `results/`.
Both are local artifacts and are ignored by Git.

Read `cant_stop_baselines/docs/EXPERIMENT_STATUS.md` before interpreting or
extending the existing results. The current evaluator alternates seats and
uses paired dice seeds; use paired analysis or independent-seed variability,
not naive independent-game confidence intervals.

## Citation and reuse

The repository is released under the MIT License. A machine-readable citation
stub is provided in [`CITATION.cff`](CITATION.cff); update it with the final
paper citation and release DOI when the project is published.
Changes to game rules, feature layouts, random-seed semantics, or evaluation
protocol should be versioned and documented as compatibility changes.
