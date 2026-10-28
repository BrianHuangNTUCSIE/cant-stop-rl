# Prior-work fidelity audit

This document records what is reproduced, adapted, or deliberately excluded
from the comparison suite. A CLI name is not by itself evidence that the
original paper or program has been reproduced.

## Status summary

Glenn–Aloi full heuristic (`glenn-aloi`) and `glenn2009_constant` are two
policy variants from the same Glenn & Aloi (2009) work. They may be compared
as policies, but are not independent prior works. `glenn2010_linear` is the
separate 2010 work.

| CLI agent | Source | Scope | Label | Important differences |
|---|---|---|---|---|
| `glenn2009_constant` | Glenn & Aloi (2009), Table 1 | pairing and Stop/Continue | `paper_reimplementation` | Uses this engine's legal-move and bank/close semantics. |
| `glenn2010_linear` | Glenn (2010), Table 6 | pairing and Stop/Continue | `paper_reimplementation` | Reimplements the published formula, not the 2010 GA search/evaluation schedule. |
| `glenn-aloi` | Glenn & Aloi heuristic | pairing and Stop/Continue | `clean_room_adaptation` | Translated formula and current engine transitions; no release binary is claimed. |
| `couto` | Medeiros, Aleixo & Lelis (2022) | pairing and Stop/Continue | `behavioral_port` | Ports the policy, not the synthesis pipeline or demonstrations. |
| `couto-ga-stop` | Same source | Stop/Continue only | `project_adaptation` | Replaces original pairing with fixed Glenn–Aloi pairing. |
| `carter-public-adapted` | Carter checkpoint and `rl_player.py` | inference | `checkpoint/feature/simulator adaptation` | Static `Prob_Table`, `columns_started`, `CarterNumpyAgent`, current rules/features; original simulator and training data are not reproduced. |
| `vmerckle_monte` | `vmerckle/CantStop-AI`, commit `25be22910e3bfc5c351f38ff5461602d95dacd0a` | root pairing + stopping | `clean_room_behavioral_port` | Root-only flat Monte Carlo and random full-game rollouts. |
| `vmerckle_monte_uct` | Same source | root pairing + stopping | `clean_room_behavioral_port` | Root-only UCB1 bandit, **not** full-tree MCTS. |
| `mc`, `mc-strong`, `mc-full-game` | This project | stopping baselines | `project_baseline` | Extensions, not external reproductions. |

## Behavioral checks

`tests/test_cant_stop.py` checks the Glenn 2009 move table (including column 7
weight 3 and marker penalty 6), the Glenn 2010 Table 6 formula, differing Glenn
and Couto stop decisions, top-reaching neutral behavior, duplicate sums, the
Carter feature width/order, and seeded/legal behavior of both vmerckle agents.
These are parity probes, not a claim that the original simulators are
identical; unresolved simulator conventions remain adaptation gaps.

Carter remains a checkpoint/feature/simulator adaptation, not a faithful
reproduction. Its low win rate after translation reflects adaptation mismatch
and must not be interpreted as evidence that Carter's original-environment
agent is weak.

## Search methods considered but not claimed

Lanctot et al. (2013), *Monte Carlo \*-Minimax Search*, is not registered as a
reproduction. The paper does not provide enough Can't Stop-specific evaluator,
chance sampling, stopping, and tuned-parameter detail to establish parity with
the reported agent. We therefore do not label the root-only UCT agent as
Star1SS, Star2SS, or full MCTS.

`mc-full-game` performs independent full-game rollouts but has no tree,
transposition table, progressive widening, or adversarial backup. It remains a
clearly named project baseline.

## Reproducibility requirements

Comparison runs record source URL/revision, adaptation label, pairing and
stopping scope, rollout budget, per-action sampling, runtime, independent base
seed, alternating seats, and raw game records. Seat-swapped games share a dice
seed, so paired bootstrap or between-seed variation is required for inference.

## Sources

* Glenn & Aloi (2009), *A Generalized Heuristic for Can't Stop*, FLAIRS.
* James Glenn (2010), *Optimizing Genetic Algorithm Parameters for a Stochastic
  Game*, ICEC/IJCCI, pp. 199–206, DOI
  [10.5220/0003079101990206](https://doi.org/10.5220/0003079101990206).
* Medeiros, Aleixo & Lelis (2022), AAAI, DOI
  [10.1609/aaai.v36i7.20744](https://doi.org/10.1609/aaai.v36i7.20744).
* Lanctot, Saffidine, Veness, Archibald & Winands (2013), IJCAI,
  [paper](https://www.ijcai.org/Proceedings/13/Papers/093.pdf).
