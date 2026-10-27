# Reproducibility Quick Start

These commands verify a clean checkout without requiring the optional ML/RL
dependencies.

## Install the core package

```bash
python3 -m pip install --no-deps -e .
```

Alternatively, `make install` runs the same core installation command and
`make verify` performs installation, tests, and the smoke test together.

The core package has no runtime dependencies. Install the optional extras only
for the corresponding experiments:

```bash
python3 -m pip install -e '.[ml]'
python3 -m pip install -e '.[rl]'
```

## Run verification

```bash
cd cant_stop_baselines
python3 -m unittest discover -s tests -t . -v
cd ..
cant-stop --agent-a rule28-smart --agent-b random --games 3 --seed 42 --json
```

The smoke test should report three games, seed `42`, and two wins for
`rule28-smart` with the current rules and implementation. The complete suite
also tests optional PyTorch components when PyTorch is installed.

## Reproduce an experiment

Before running an existing experiment:

1. Read `cant_stop_baselines/docs/EXPERIMENT_STATUS.md`.
2. Record the repository commit with `git rev-parse HEAD`.
3. Record the Python version and installed dependencies.
4. Use the experiment's documented seed, opponent configuration, game count,
   and model checkpoint.
5. Preserve the generated JSON metadata and raw game-level CSV records.

Run directories and checkpoints are intentionally ignored by Git. Publish
selected outputs as release assets or in an archival repository together with
their producing commit, configuration, and checksums.

## Evaluation caveat

The evaluator alternates seats and paired games can share dice seeds. Treat
those games as paired observations; use the paired bootstrap or variability
across independent training/evaluation seeds rather than naive independent-game
confidence intervals.
