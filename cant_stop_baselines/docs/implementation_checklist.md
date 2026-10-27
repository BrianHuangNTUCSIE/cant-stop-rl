# Agent Implementation Checklist

Phase 0 is complete only when the following are explicit. Items marked **blocked** require a source decision before implementation.

## Phase 0 — audit only

- [x] Read `AGENTS.md`, project guide, project spec pointer, README, architecture notes, experiment status, engine, agents, evaluator, DQN path, and tests.
- [x] Record actual engine and agent APIs in `engine_integration_audit.md`.
- [x] Verify accessible primary sources and public repository layouts/licenses in `source_audit.md`.
- [x] Confirm existing defaults (`random`, `rule28-smart`, `mc`, `ddqn` path) are to remain unchanged.
- [x] Confirm no local Medeiros or Carter PDFs are present.
- [x] Confirm 2009 Constant table against the primary PDF.
- [x] Resolve the guide-vs-paper discrepancy: implement the verified paper Table 6 variant and document the guide transcription as unresolved/adaptation-only.
- [x] Restore/document `docs/THIRD_PARTY_NOTICES.md`; no unlicensed source is redistributed.

## Phase 1 — deterministic heuristics

- [x] Decide and document names/classification: `glenn2009_constant` and the verified Table 6 `glenn2010_linear` variant.
- [x] Implement as thin `Agent` adapters in `cant_stop_core/agents.py`; do not edit `Game`.
- [x] Separate pair selection from Stop/Continue policy; vmerckle evaluates pair/action combinations at the root.
- [x] Implement complete paper scoring: symmetry, progress denominator, marker penalties, difficulty bonuses, threshold, top/blocked columns, winning bank, invalid-pair behavior, and deterministic tie-breaking.
- [x] Add hand-calculated equation tests and winning/boundary coverage; existing engine tests cover legal pairings, bust, banking, and wins.
- [x] Register names through `AGENT_NAMES` and `make_agent`; ensure old names produce unchanged behavior.
- [x] Add an evaluation-only runner for 100-game smoke evaluations against registered baselines with raw records/config metadata; DDQN remains optional when a checkpoint and PyTorch are available.
- [x] Add dependency-free paired bootstrap/sign-flip analysis grouped by evaluator dice seed.
- [x] Support independent evaluation replicates with deterministic, non-overlapping seed batches.
- [x] Aggregate replicate margins for descriptive between-seed variability.

## Phase 2 — sampling search

- [x] Extract full `vmerckle` simulation/rollout semantics and license status.
- [x] Implement explicit rollout budget and private RNG/state simulation.
- [x] Keep root-UCB distinct from full UCT/MCTS in names and docs.
- [x] Test invalid/low budgets and seeded results; engine tests cover legal actions and terminal transitions.

## Phase 3 — programmatic strategies

- [x] Inspect Medeiros DSL grammar, AST/interpreter, SA/UCT operators, demonstrations, outputs, and evaluation.
- [x] Add attribution/license notice; no source code was copied.
- [x] Provide a dependency-free JSON AST inference adapter; synthesis commands remain separate and unimplemented.
- [x] Mark absent serialized champions and substitute programs as `unavailable`/`adaptation`.

## Phase 4 — Carter/deeper search

- [x] Verify Carter's public feature order/scaling/perspective and model architecture from source; document engine-mismatch limitations.
- [x] Add an optional, path-supplied `CarterKerasAgent` loader; do not register or redistribute the checkpoint until TensorFlow loading and cross-engine validation are available.
- [x] Keep existing DDQN separate; do not call Double DQN “Carter’s model.”
- [ ] Implement ExpSS/Star/DPW only after paper-specific missing details and classification are recorded.

## Phase 5 — evaluation

- [x] Use same engine/rules, alternating seats, separate RNG streams, and disjoint evaluation seeds.
- [x] Report game count, draws, CI/inference method, seeds, pairing, search budget, latency, raw records, and provenance in the evaluation runner; formal results remain to be generated.
- [x] Use paired bootstrap or between-seed variability for the evaluator’s shared-seed seat pairs; do not treat Wilson intervals as independent-game evidence.
- [ ] Produce the required comparison matrix only after each agent passes legality, determinism, smoke, and provenance acceptance criteria.
