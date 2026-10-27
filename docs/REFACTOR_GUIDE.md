# Refactor and Publication Guide

## Purpose

The repository is being converted from a course-workspace layout into a public
research repository. The public contract is a reusable *Can't Stop* engine;
the paper experiments are clients of that engine, not part of its rules.

This guide records the completed compatibility-first migration. The package
boundary is now stable; experiment commands remain available, but the old
source-layout compatibility files have been removed.

## Current boundary

The reusable implementation now lives under `src/`:

| Layer | Responsibility | Allowed dependencies |
|---|---|---|
| `src/cant_stop/model.py`, `constants.py` | Immutable state, moves, board constants | Standard library |
| `src/cant_stop/rules.py` | Pure legal-move and scoring functions | Core model/constants |
| `src/cant_stop/game.py` | Seeded mutable state machine | Core rules/model |
| `src/cant_stop/agents.py` | Policy adapters and heuristics | Core state/rules |
| `src/cant_stop/rl_env.py` | Learner-facing environment | Core engine |
| `src/cant_stop/evaluation.py` | Game execution and metrics | Core engine and agents |
| `src/cant_stop/dqn.py`, `ppo.py` | Optional learning implementations | PyTorch |
| `src/learners/`, `src/rollout/` | Experiment orchestration support | Core plus optional dependencies |

The engine must not import training scripts, report code, plotting libraries,
or experiment configuration. Agents must not mutate agent-facing snapshots.

## Target public layout

The package now uses the conventional `src/cant_stop/` layout. The old
`cant_stop.py` compatibility facade has been removed; use the installed
`cant-stop` command or import `cant_stop` directly.

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
- Migrate the reusable package into `src/` and remove the nested repository
  metadata after the migration is verified.

### Phase 2: Engine contract (implemented)

- Define versioned `Game`, `State`, `Move`, and transition protocols.
- Document legal actions and phase transitions.
- Add deterministic replay fixtures using explicit dice and RNG streams.
- Test marker limits, duplicate sums, completed columns, bust/bank behavior,
  terminal states, and seeded reproducibility.

### Phase 3: Agent and environment contracts (implemented)

- Use one policy interface for move selection and Stop/Continue decisions.
- Keep dice pairing, stop policy, state encoding, reward, and opponent factory
  independently configurable.
- Ensure an agent can be registered without editing engine code.
- Keep optional PyTorch and scikit-learn dependencies out of core imports.

### Phase 4: Configuration-driven experiments (in progress)

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
