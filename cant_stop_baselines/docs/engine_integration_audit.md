# Engine Integration Audit

**Audit date:** 2026-10-27  
**Scope:** Phase 0 of `CANT_STOP_AGENT_IMPLEMENTATION_GUIDE.md`

## Existing architecture

| Responsibility | File | Actual API / finding |
|---|---|---|
| Immutable agent state | `cant_stop_core/model.py` | `State(permanent, owner, active, current_player, rolls_this_turn)`; `Move(columns)`; `State.features()` returns 58 raw, 60 enhanced, 68 domain, or 80 strategic values. |
| Rules | `cant_stop_core/rules.py` | `legal_moves(dice, permanent, owner, active)`, `advance_active(...)`, `bust_probability(state)`, `rule28_points(state)`, `pair_score(...)`. Pure functions. |
| Mutable engine | `cant_stop_core/game.py` | `Game.roll(rng, dice=None)`, `apply_move(move)`, `continue_turn()`, `stop()`, `snapshot()`. Phases: `roll`, `pair`, `decision`, `done`. No policy code. |
| Agent contract | `cant_stop_core/agents.py` | `Agent.choose_move(state, moves, rng) -> Move` and `Agent.should_stop(state, rng) -> bool`. This is the adapter contract to preserve. |
| Agent registry | `cant_stop_core/agents.py` | `AGENT_NAMES` and `make_agent(name, mc_rollouts=64, mc_future_roll_limit=40)`. CLI choices come from `AGENT_NAMES`. |
| Evaluation | `cant_stop_core/evaluation.py` | `play_game(agents, seed, max_total_turns=2000)` and `evaluate(...)`. Alternates seats; game `i` uses `seed + i//2`, so seat-swapped games share dice seeds. Dice RNG and two seat-specific policy RNGs are separate. |
| CLI | `cant_stop_core/cli.py`, `cant_stop.py` | `--agent-a`, `--agent-b`, `--games`, `--seed`, `--mc-rollouts`, `--max-turns`, JSON/CSV output. |
| DQN | `cant_stop_core/dqn.py`, `scripts/train_dqn.py` | `DQNAgent.from_checkpoint(path, device=...)`; existing two-action Stop/Continue and learned-pairing modes. Do not rename or replace existing DDQN. |
| RL environment | `cant_stop_core/rl_env.py` | Dependency-free Stop/Continue environment; joint pairing mode is already supported by the training path. |
| Tests | `cant_stop_baselines/tests/` | Rule, transition, evaluation, RL, DQN, classical ML, PPO, native parity, and migration tests. |

## Rules and state invariants observed

- Two players; columns 2--12; heights are `3,5,7,9,11,13,11,9,7,5,3`.
- A turn can have at most three active columns. If both sums in a pairing are legal, both are used; when only one runner is available, legal single-sum alternatives are exposed.
- Temporary progress lives in `Game.active` as absolute positions. `stop()` banks it and closes columns only at the top. A bust clears temporary progress and switches player without changing permanent progress.
- A player wins after banking a third owned column. A temporary third completed column is not a win until banked; agents explicitly check this where needed.
- `Game.roll()` owns dice generation through the caller-provided `random.Random`. Agents only receive a separate policy RNG.
- The engine does not expose a general clone/simulate method. `State` is immutable, and search agents currently construct a local `Game` only for the existing full-game MC adapter (`FullGameMonteCarloAgent._copy_game`). New search should use a private simulation adapter or explicit state clone without mutating the live game.

## Existing agents and configuration

Registered names currently are:

`random`, `rule28`, `rule28-smart`, `rule28-ga-pure`, `mc`, `mc-full-game`, `mc-strong`, `vmerckle_monte`, `vmerckle_monte_uct`, `glenn-aloi`, `glenn2009_constant`, `glenn2010_linear`, `couto`, `couto-ga-stop`.

- `random`: random legal pairing and probabilistic stop (`0.3` default).
- `rule28` / `rule28-smart`: Rule-of-28 continuation with classic or GA pairing as configured.
- `mc`: one-turn approximate MC for Stop/Continue, fixed GA pairing and Rule28-smart rollout policy; explicitly not full-game MCTS.
- `mc-full-game`: full-game rollout policy, separately named.
- `glenn-aloi`: existing released Sketch-SA(O)-derived/adapted policy, not the requested separately auditable `glenn2009_constant` or `glenn2010_linear` implementations.
- `couto` / `couto-ga-stop`: existing Sketch-SA(O) policy and a fixed-GA pairing adaptation.
- `ddqn` is a saved-model/evaluation path rather than a built-in name in `AGENT_NAMES`; preserve its current checkpoint and loader behavior.

## Compatibility gaps for the guide

1. The current protocol calls the pairing decision `choose_move`, not `choose_pair`; use a thin adapter or the existing protocol rather than changing `Game`.
2. The current `Agent` protocol has no explicit opponent-facing search context, clone, chance sampler, or terminal utility. Add search helpers outside `Game` only when a search phase begins.
3. There are no registered `glenn2009_constant` or `glenn2010_linear` names. The existing `glenn-aloi` implementation has a fixed threshold-29 policy and GA pairing, but must not be treated as proof of either paper champion.
4. `docs/THIRD_PARTY_NOTICES.md` is referenced by `README.md` but is absent in the working tree. This must be repaired or the README reference corrected before redistributing any ported source.
5. The README documents `python` commands although the current environment provides `python3`; Phase 0 verification should use `python3`.

## Planned integration boundary

Implement new agents as classes satisfying the existing two-method `Agent` protocol. Register them only through `AGENT_NAMES`/`make_agent`, and add CLI/evaluation support through the existing factory. Do not modify `Game`, `legal_moves`, feature ordering, RNG ownership, or existing agent defaults. Pairing and Stop/Continue behavior must be separately named in documentation and tests.
