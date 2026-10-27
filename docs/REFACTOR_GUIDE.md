# Refactor and Publication Guide

## Purpose

The repository is being converted from a course-workspace layout into a public
research repository. The public contract is a reusable *Can't Stop* engine;
the paper experiments are clients of that engine, not part of its rules.

This guide deliberately uses a compatibility-first migration. Existing import
paths and experiment commands remain available while the package boundary is
stabilized. Large moves should happen only after tests and reproduction checks
pass.

## Current boundary

The current implementation lives under `cant_stop_baselines/`:

| Layer | Responsibility | Allowed dependencies |
|---|---|---|
| `cant_stop_core/model.py`, `constants.py` | Immutable state, moves, board constants | Standard library |
| `cant_stop_core/rules.py` | Pure legal-move and scoring functions | Core model/constants |
| `cant_stop_core/game.py` | Seeded mutable state machine | Core rules/model |
| `cant_stop_core/agents.py` | Policy adapters and heuristics | Core state/rules |
| `cant_stop_core/rl_env.py` | Learner-facing environment | Core engine |
| `cant_stop_core/evaluation.py` | Game execution and metrics | Core engine and agents |
| `cant_stop_core/dqn.py`, `ppo.py` | Optional learning implementations | PyTorch |
| `scripts/`, `learners/`, `rollout/` | Experiment orchestration | Core plus optional dependencies |

The engine must not import training scripts, report code, plotting libraries,
or experiment configuration. Agents must not mutate agent-facing snapshots.

## Target public layout

The package will eventually be moved to a conventional `src/cant_stop/`
layout. Until then, `pyproject.toml` installs the existing package from
`cant_stop_baselines/` and the old `cant_stop.py` facade remains supported.

```text
src/cant_stop/
  engine/          # rules, state, transitions, seeded game
  agents/          # policy interface and implementations
  environments/    # RL adapters
  evaluation/      # runners, metrics, paired statistics
  io/              # configs, checkpoints, result schemas
experiments/
  configs/         # reviewed, complete experiment specifications
  scripts/         # thin command-line entry points
tests/
docs/
results/           # generated and normally ignored
```

## Migration phases

### Phase 1: Repository hygiene (implemented)

- Add root packaging metadata and a public README.
- Add this guide as the architectural source for the migration.
- Add root ignore rules for environments, caches, checkpoints, logs, and
  generated report intermediates.
- Keep CSV/JSON experiment evidence available locally while excluding generated
  run directories from future commits.
- Preserve the existing nested Git history as `.git-legacy-backup/` while the
  root repository becomes the official repository.

### Phase 2: Engine contract

- Define versioned `Game`, `State`, `Move`, and transition protocols.
- Document legal actions and phase transitions.
- Add deterministic replay fixtures using explicit dice and RNG streams.
- Test marker limits, duplicate sums, completed columns, bust/bank behavior,
  terminal states, and seeded reproducibility.

### Phase 3: Agent and environment contracts

- Use one policy interface for move selection and Stop/Continue decisions.
- Keep dice pairing, stop policy, state encoding, reward, and opponent factory
  independently configurable.
- Ensure an agent can be registered without editing engine code.
- Keep optional PyTorch and scikit-learn dependencies out of core imports.

### Phase 4: Configuration-driven experiments

- Replace shell-script-only settings with validated YAML or JSON configs.
- Record rules version, feature schema, algorithm, hyperparameters, model path,
  training/evaluation seeds, opponent mix, and game count.
- Write raw game records as CSV and run metadata as JSON.
- Generate plots and paper tables from those raw records in one command.

### Phase 5: Statistical evaluation

- Preserve alternating seats and paired dice-seed metadata.
- Report paired bootstrap intervals or independent-seed variability correctly.
- Separate smoke tests from report-scale evaluations.
- Never select or report only the best training seed without showing the seed
  protocol and variability.

### Phase 6: Release engineering

- Add CI for unit tests, packaging, and a short deterministic reproduction.
- Add type/lint checks and documentation checks.
- Publish a versioned release with source, configs, selected CSV evidence,
  checkpoints where licensing and size permit, and a DOI.
- Add `LICENSE`, `CITATION.cff`, and a changelog before public release.

## Artifact policy

### Keep and publish deliberately

- Small CSV files needed for plots or statistical analysis.
- JSON configuration and summary metadata.
- Dataset documentation and checksums.
- Selected model checkpoints tied to a release.
- Source code, tests, documentation, and paper source.

### Do not commit

- `.log` files and terminal transcripts.
- `.DS_Store`, `__pycache__`, virtual environments, and tool caches.
- LaTeX `.aux`, `.out`, `.toc`, and similar intermediates.
- Every intermediate checkpoint from a long training run.
- Unlabelled or duplicated result directories.

Large evidence files should move to a release asset or archival repository,
with a manifest containing checksums and the exact producing configuration.

## Compatibility rules

Any change to board heights, action legality, feature ordering, reward meaning,
RNG stream derivation, seat alternation, or terminal handling is a protocol
change. It requires:

1. A version number or migration note.
2. Updated fixtures and tests.
3. A changelog entry.
4. Explicit labeling of results produced under each version.

Published methods must be labeled as faithful reproductions, adaptations, or
inspired baselines. The repository must not imply that a published win rate was
reproduced unless the implementation and evaluation protocol support that claim.

## Definition of done

The refactor is complete when a clean clone can:

1. Install the core engine without ML dependencies.
2. Run the complete engine test suite.
3. Run a deterministic two-agent example.
4. Train an optional model using a checked-in configuration.
5. Evaluate it against named baselines and produce raw CSV records.
6. Regenerate the paper plots from those records.
7. Identify the source commit, configuration, seeds, and dependencies for every
   reported number.
