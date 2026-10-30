# Report evidence guide

## Central question and application

Can learned Stop/Continue strategies outperform traditional heuristics with
fixed Glenn–Aloi pairing? How do model complexity and collected state coverage
change complete-game performance? The browser application lets players practice
against learned opponents. The controlled simulator supports the quantitative
ML comparison. Fixed pairing isolates one strategic decision, not all game play.

## Keep evidence sets separate

| Evidence set | Models / protocol | Supported conclusion |
|---|---|---|
| Common held-out, seed 1000454 | PPO-451, reference DDQN, original Ridge; 3,000 games per matchup | Selected PPO checkpoint improved over these controls; see paired differences below |
| Post-merge, seed 20261103 | PPO-460 top_1, DDQN-448 best, multi-policy Ridge-642; 3,000 games, MC-64 | Current prior-agent comparison; not the same checkpoints as the earlier control study |
| Two additional training seeds | PPO 460/461, DDQN 448/449, Ridge 642/643; evaluation seed 20261101 | Limited training-seed robustness; do not infer population-wide algorithm dominance |
| Browser exports | DDQN-447 best, PPO-460 best, Ridge-642 | Local JavaScript deployment; not a measured reproduction of the Python main table |

In the common held-out study, PPO minus DDQN was +3.2 percentage points
against Rule28-smart (paired 95% interval [1.2, 5.3]) and +2.7 against MC
([0.6, 5.0]). PPO minus original Ridge was +4.5 ([2.2, 6.7]) and +4.3
([1.9, 6.6]). These support checkpoint-specific differences, not a general
conclusion that neural policies outperform multi-policy Ridge.

See [final evidence](final_results/FINAL_EVIDENCE.md),
[consistency audit](final_results/CONSISTENCY_AUDIT.md), and the
[experiment record](EXPERIMENT_STATUS.md). Seeds and archive labels are
identifiers; do not infer execution dates from their digits.

## Data collection and curation

The simulator produces training trajectories and project-designed held-out
games. Explain this interpretation of self-collected evaluation data explicitly.
A supervised record is a decision state with source-game identity, features,
forced-action rollout estimates, and a continuation-margin target. The target
is an approximation under the rollout continuation policy, not optimal play.

The classical trainer groups records by source `game_id` before splitting
70/15/15. Hyperparameters use validation; the selected estimator is refit on
train plus validation before the unseen supervised test split is scored.
Complete-game evaluation is a separate efficacy measurement. RL validation
selects checkpoints; final game seeds must remain separate from that selection.

The multi-policy dataset has 10,000 states from Rule28-smart, MC, Ridge,
PPO-451, and DDQN. These are state-collection policies; all labels use the
common forced-action counterfactual procedure and Rule28-smart rollouts.
Before submission, extract exact per-policy counts, unique source-game counts,
rollout budgets, filtering, and split counts from the retained dataset and its
metadata. Do not invent equal proportions or derive unique-game counts from
state counts. Publish the retained data or a documented collection recipe and
usable data location; the sample CSV is not experimental evidence.

## Analysis worth emphasizing

- Multi-policy Ridge improved over original Ridge by +3.3 points against
  Rule28-smart ([1.0, 5.6]) and +4.2 against MC ([2.0, 6.7]) on seed 20261030.
- Multi-policy DNN achieved 32.3% / 28.7% against Rule28-smart / MC on seed
  20261029. Broader state coverage alone did not yield a successful DNN.
- Learned pairing lost to fixed-GA PPO by 14.1 / 10.9 points in the paired
  diagnostic. More learned components did not automatically improve play.

Keep comparisons within their evaluation batch. Overfitting, noisy labels,
and distribution shift remain hypotheses unless additional diagnostics test
them. Do not launch another long experiment solely to decorate the report.

## Statistical and comparison boundaries

Resample shared-dice, seat-swapped pairs, not independent games. Report paired
intervals for differences when claiming improvements. Distinguish variation
across games from variation across independently trained models; more games do
not replace training seeds. Report draws, truncations, and the win-rate
denominator from the actual evaluator. State whether multiple comparisons
were adjusted; avoid silently treating nominal intervals as family-wide tests.

Carter is a checkpoint/feature/simulator adaptation. Its low Random calibration
cannot establish that the original method is weak. Root-only UCB1 is not full
MCTS. End-to-end comparisons with agent-specific pairing differ from the
scope-matched Stop/Continue comparison. No exhaustive open-source leaderboard
or universal best-agent claim has been established.

## Submission completion

Use the professor's report template and re-check the current specification.
Include actual hyperparameters, one or two real state/action examples,
acknowledgements, individual contributions, video link, and valid code/data
links. Verify public artifact access before submitting. The current course
specification prohibits AI-generated slides and video; students must create
those deliverables themselves and keep the video within six minutes.
