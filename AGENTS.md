# CS 441 Can't Stop Final Project

## Source of truth and status

- `spec.pdf` is the professor's Fall 2026 assignment. Re-check it before submission: it says it was under design review and details could change before Nov. 1.
- Current code in `cant_stop_baselines/` includes the baseline simulator, a dependency-free `StopContinueEnv`, PyTorch DQN/Double DQN agents, training/evaluation pipelines, and classical-ML experiments. `cant_stop.py` is a compatibility/CLI facade; `cant_stop_core/` separates constants, models, pure rules, game state, agents, evaluation, the RL environment, DQN components, and CLI. Python entry scripts and shell launchers live in `cant_stop_baselines/scripts/` (invoke them from `cant_stop_baselines/`, e.g. `python scripts/train_dqn.py`), tests in `cant_stop_baselines/tests/`, documentation in `cant_stop_baselines/docs/`, and sample data in `cant_stop_baselines/data/`.
- `cant_stop_baselines/docs/EXPERIMENT_STATUS.md` is the source of truth for completed experiment results, current model-selection decisions, and the next unresolved question. Read it before proposing or launching another experiment.
- `cant_stop_baselines.zip` is an archive, not the working source. Do not edit it or treat `example_results.csv` as experimental evidence.
- This file labels course requirements separately from project decisions. Do not present a project choice as professor-mandated.

## Professor-stated requirements

- Build a proof-of-concept application using ML to solve a motivated problem. The ML component is the emphasis; a minimal interface is acceptable.
- Explain the application, problem, intended user, concrete use case/benefit, and ML formulation (inputs, outputs, models/techniques).
- Create and explain the data collection/annotation pipeline, sources, curation, and dataset organization. Training may use any resource, but final model evaluation must use data we collect and an evaluation set we design for this project.
- Do not use a benchmark, Kaggle, or another pre-curated ML dataset for final evaluation. Ordinary online source material is allowed if it was not curated as an ML dataset.
- Compare at least a baseline and one proposed ML approach; vary key design decisions and document design rationale, training procedure, and hyperparameters.
- Evaluation must be quantitative. Tabulate/plot results, justify conclusions, and determine whether model differences are statistically significant.
- Final deliverables are the completed report template and a 5–6 minute video; never exceed 6 minutes. AI may **not** generate the video or slides.
- Video allocation/content: application and motivation (~1 min); ML problem/approach, 1–2 input/output examples, data collection/curation, baseline and proposed approaches (~2–3 min); key results/conclusions and support for the final approach (~1–2 min). An application demo may extend this by up to 1 minute while remaining within 6 minutes.
- Submit one report per group through Gradescope and add all group members. Mark pages for graded sections. Include acknowledgements, individual contributions, the video link, and links to code and data (or describe confidential data). A valid code link is mandatory.
- Deadline in the current PDF: **Sunday, Dec. 13, 2026 at 11:59 PM**. No late days may be used for the project or its components.
- Proposal points: practice proposal 20, peer review 10, full proposal 20. Final submission offers 165 points against a 150-point target; points beyond the overall 200-point target count as XP.
- Final grading: Problem Description 15 (goal/motivation 5, use case 5, ML formulation 5); ML Approach 40 (design 5, training/hyperparameters 5, quality/innovation 15 average/30 excellent); Data 40 (sources/collection 5, curation/organization 5, quality/effort/innovation 15 average/30 excellent); Experiments & Analysis 40 (presentation 10, sound conclusions 10, quality/thoroughness/innovation 10 average/20 excellent); Video 30 (required content 10, clarity/quality 10 average/20 excellent).
- Checklist penalties include unmet project requirements (e.g. evaluation data from an ML source) -20, missing group attribution/contributions -10, and missing code or data link -30.

## Our research plan (not course mandates)

- Research question: **Can RL agents learn better Stop/Continue strategies than traditional heuristic-based agents in Can't Stop?**
- Use the standard two-player game. Initially hold dice-pair selection fixed so DQN/Double DQN comparisons isolate the Stop/Continue policy; keep pairing interchangeable for later experiments.
- Pairing candidates include the published Glenn & Aloi constant move weights. Baselines are random, pure Rule of 28, a clearly named/documented smarter variant, and an approximate Monte Carlo rollout agent if feasible.
- Compare raw fixed-length state features against domain features such as exact next-roll bust probability and at-risk progress. Keep algorithm, state encoding, reward, and pairing policy independently swappable.
- Evaluate agents under identical rules and pair-selection assumptions, with deterministic seeds, alternated starting seats, multiple independent training/evaluation seeds, uncertainty estimates, and ablations. Report win rate, bust rate, and relevant turn/roll statistics; save raw per-game results and complete experiment configs.
- Simulator-generated games are our proposed data collection process. In the report, explain how generated training and held-out evaluation games satisfy the course's collection/evaluation requirement rather than silently assuming this interpretation.
- Do not claim any heuristic or learned method is effective without experimental evidence. Distinguish reproduced published methods from adaptations and novel extensions.

## Architecture and invariants

- Keep game rules/state transitions independent of agents, training, evaluation, and experiment configuration. Do not put policy logic into the engine.
- Preserve legal-move generation, deterministic RNG injection, immutable agent-facing snapshots, and an explicit fixed-length state encoding. Avoid assumptions that block alternate policies, rewards, encodings, or pairing methods.
- Standard rules currently encoded: columns 2–12 have heights `3,5,7,9,11,13,11,9,7,5,3`; at most 3 active columns; use both sums when jointly legal; bust loses only unbanked turn progress; a column closes only when banked; 3 closed columns wins.
- The current training-facing state machine is `Game.snapshot()`, `roll()`, `apply_move()`, `stop()`, and `continue_turn()` with phases `roll`, `pair`, `decision`, `done`. Stop/Continue transitions must account for busts, opponent turns, and terminal rewards.
- `State.features()` has 58 values. `features(enhanced=True)` has 60, adding exact bust probability (all 1,296 rolls) and normalized at-risk progress. Do not repeat the obsolete 57-feature count.
- Current evaluator alternates seats and gives each seat independent deterministic agent RNG streams. Consecutive seat-swapped games share a dice seed, so games within a pair are not independent; its Wilson intervals are descriptive only. Use paired bootstrap/inference or between-seed variability for report claims.
- Current `mc` is an approximate remainder-of-turn rollout using GA pairing and smart Rule of 28, not full-game MCTS. `rule28-smart` is a local hybrid, not Carter's full smart Rule of 28 or Glenn & Aloi's full optimized agent.
- Glenn & Aloi Table 1 move weights for columns 2–12 are `7,0,2,0,4,3,4,0,2,0,7`; the new-marker penalty is 6. In particular, column 7 has weight 3.

## Development and verification

- Use Python and PyTorch for new RL work. Prefer typed, small interfaces and straightforward modules over framework-heavy abstractions.
- Run the existing suite from `cant_stop_baselines/` with `python3 -m unittest discover -s tests -t . -v`. This environment has no `python` executable; all 61 tests passed on 2026-10-25 (11 skipped for optional third-party dependencies).
- Focused test: `python3 -m unittest -v tests.test_cant_stop.TestCantStop.test_pairings` (replace the final test name as needed).
- Add tests for every rule or state-transition change, especially pairing legality, marker limits, duplicate sums, completed columns, bust/bank behavior, terminal states, and seeded reproducibility.
- Preserve separate dice and agent randomness and record seeds, dependency versions, hyperparameters, checkpoints, configs, and raw metrics needed to reproduce every reported result.
- The baseline currently requires Python 3.10+ and has no third-party dependencies. No root environment/packaging, lint, formatter, or CI configuration exists yet; do not invent commands for them.
