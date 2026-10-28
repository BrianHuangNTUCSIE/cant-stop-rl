# Experiment Status

Updated 2026-10-27.

## Experiment infrastructure readiness

The in-process Python reference training path is ready for new controlled
experiments. A 20-episode smoke run with two environments completed training,
validation, checkpoint writing, and metrics export on 2026-10-26. During that
smoke run, a training-loop variable shadowing bug was fixed: the learner seat
index now uses `learner_player` and no longer overwrites the `DQNLearner`
instance. The regression suite includes a short subprocess training test.

Use `--backend python` for formal runs. The native C++ work remains an optional
parity prototype and is not part of the experiment backend.

## Current decision

The retained RL reference remains the completed MC-counterfactual DDQN:

```text
runs/ddqn_mc_counterfactual_seed447/best.pt
```

The Ridge-opponent robustness experiment was stopped early and is not a
candidate for promotion.

## PPO scratch feasibility pilot

The 20,000-episode PPO pilot completed from scratch with seed `450`:

```text
runs/ppo_scratch_pilot_seed450/
```

It used the 68 domain features, fixed GA pairing, terminal-only rewards, a
128x128 actor-critic, and the same training-opponent mixture as the retained
DDQN (35% `rule28-smart`, 35% MC, and 10% each pure-GA Rule 28, classic Rule
28, and random). Training used MC-16 with a future-roll limit of 20 and eight
multiprocess environments. This was a single-seed feasibility run, not a
formal algorithm comparison.

Validation results were:

| Episode | Rule28-smart | MC | Minimum |
|---:|---:|---:|---:|
| 10,073 | 61.2% | 56.2% | 56.2% |
| 20,000 | 62.0% | 53.4% | 53.4% |

The episode-10,073 checkpoint was retained as `best.pt`. The minimum
validation rate declined by 2.8 percentage points by the end, primarily from
the MC matchup, so checkpoint selection matters even in this short run.

The automatic final-checkpoint evaluation used 1,000 games per opponent and
MC-32:

| Final PPO opponent | Win rate | Descriptive Wilson 95% CI | Busts/turn |
|---|---:|---:|---:|
| Rule28-smart | 62.5% | 59.5%--65.4% | 0.383 |
| MC | 52.8% | 49.7%--55.9% | 0.360 |

This passes the predeclared feasibility gate: the minimum exceeded 40%, and
PPO learned a competitive policy from scratch. It does **not** establish that
PPO beats the retained DDQN. The run has only one training seed, the final
evaluation evaluated `final.pt` rather than `best.pt`, MC-32 differs from the
formal robustness setup, and no raw per-game records or paired comparison with
the DDQN were saved. The apparent differences from the retained DDQN's 60.9%
versus Rule28-smart and 53.9% versus MC therefore must not be interpreted as a
significant improvement.

### PPO common held-out checkpoint evaluation

PPO checkpoint evaluation support was added to `scripts/evaluate_ml.py`, and
the PPO best and final checkpoints were compared with the retained DDQN using
the same fresh seed `700450`, 3,000 games per matchup, MC-64, alternated seats,
and 5,000 paired-bootstrap samples. Raw games and the complete summary are in:

```text
runs/ppo_common_eval_seed700450/
```

| Checkpoint | Rule28-smart | MC | Minimum |
|---|---:|---:|---:|
| PPO best (episode 10,073) | 63.1% | 54.5% | 54.5% |
| PPO final (episode 20,000) | 61.6% | 52.7% | 52.7% |
| Retained DDQN | 61.8% | 53.5% | 53.5% |

The paired differences for PPO best minus DDQN were +1.3 percentage points
against Rule28-smart (95% interval -0.8 to +3.4) and +1.0 point against MC
(95% interval -1.4 to +3.3). Neither interval excludes zero. PPO best also
outscored PPO final by 1.5 points against Rule28-smart and 1.7 against MC, but
those differences were not significant either. Its bust rates were 0.427
against Rule28-smart and 0.410 against MC, versus 0.437 and 0.429 for DDQN on
the same evaluation.

Decision: PPO passes the continuation gate and is competitive with DDQN on
both primary opponents, but this single training seed does not establish an
improvement. Retain the MC-counterfactual DDQN as the reference. The next PPO
experiment may proceed to longer, multi-seed training; promotion requires
between-seed evidence and a fresh common held-out evaluation rather than the
point estimates from this selection run alone.

### MC-focused PPO experiment completed

The seed-`451` run completed 100,000 episodes from scratch using 60% MC and 40%
`rule28-smart` training opponents. All other settings matched the successful
pilot: 512-step rollouts, four PPO epochs, learning rate `1e-4`, a 128x128
network, MC-16 training with future-roll limit 20, terminal-only reward, domain
features, and fixed GA pairing. Artifacts are in:

```text
runs/ppo_mc_focused_seed451/
```

The three selected validation checkpoints were:

| Episode | Rule28-smart | MC | Minimum | Busts/turn | Greedy Stop |
|---:|---:|---:|---:|---:|---:|
| 95,021 | 67.2% | 60.6% | 60.6% | 0.396 | 11.5% |
| 35,095 | 61.2% | 59.2% | 59.2% | 0.418 | 10.9% |
| 30,073 | 60.4% | 58.4% | 58.4% | 0.402 | 11.6% |

The episode-95,021 result clears the desired 60% threshold against both target
opponents, but it is a checkpoint-selection result from only 500 games per
opponent and must not be treated as held-out evidence. Performance remained
checkpoint-sensitive: at episode 100,000, validation was 63.4% against
`rule28-smart` and 53.6% against MC. The automatic 3,000-game evaluation of
`final.pt` was 62.4% (descriptive Wilson interval 60.7%--64.1%) against
`rule28-smart` and 54.8% (53.0%--56.6%) against MC.

Training diagnostics show no numerical instability or catastrophic optimizer
update. Across 883 updates, all recorded metrics were finite, approximate KL
never exceeded 0.0062, clip fraction never exceeded 3.45%, and the final 100
updates averaged 0.516 value explained variance. Entropy declined from a mean
of 0.207 over the first 100 updates to 0.101 over the last 100, consistent with
a more deterministic policy. The large validation oscillations despite stable
update diagnostics reinforce the need for held-out checkpoint comparison.

Decision: retain the DDQN reference for now. Evaluate the episode-95,021
checkpoint (`top_1.pt`) against the seed-450 PPO best and retained DDQN using a
fresh common seed, 3,000 games per matchup, MC-64, raw game records, and paired
bootstrap inference. This single training seed can identify a candidate but
cannot establish between-seed reliability.

### MC-focused PPO common held-out evaluation

The requested common evaluation completed with seed `800451`, 3,000 games per
matchup, MC-64, alternated seats, raw game records, and 5,000 paired-bootstrap
samples. Artifacts are in:

```text
runs/ppo_mc_focused_common_eval_seed800451/
```

| Checkpoint | Rule28-smart | MC | Minimum |
|---|---:|---:|---:|
| PPO seed 451, episode 95,021 | 65.6% | 54.3% | 54.3% |
| PPO seed 450, episode 10,073 | 64.2% | 54.1% | 54.1% |
| Retained DDQN | 62.4% | 52.8% | 52.8% |

PPO-451 exceeded DDQN by 3.17 percentage points against `rule28-smart` (paired
95% interval 1.03 to 5.20), a nominally statistically significant result for
this fixed set of trained checkpoints; the interval was not adjusted for the
multiple reported comparisons. Its 1.53-point advantage against MC was not
significant (interval -0.77 to 3.87). PPO-451 also did not significantly beat
PPO-450: the differences were +1.43 points against `rule28-smart` (interval
-0.63 to 3.50) and +0.20 against MC (interval -2.03 to 2.47).

The selected checkpoint's paired-bootstrap win-rate interval against MC was
52.5%--56.0%, so the 60% MC validation result did not transfer to held-out
evaluation. Raising the MC training share from 35% to 60% therefore did not
provide evidence of an MC improvement over PPO-450. The seed-451 checkpoint is
the strongest point-estimate candidate and has a supported Rule28-smart gain
over DDQN, but the run still has only one training seed and does not establish
between-seed reliability or a general PPO improvement.

Decision: do not spend the next experiment solely on longer fixed-GA training
or a deeper MLP. First test whether fixed GA dice pairing is the limiting
component. A cheap diagnostic should compare this same frozen PPO checkpoint
under GA and Couto pairing on common dice seeds. If Couto helps both primary
matchups, retrain one controlled fixed-Couto PPO run; otherwise prioritize a
joint learned-pairing PPO implementation with legal action masking. Treat
network depth as a later ablation after pairing and representation are fixed.

### Frozen PPO pairing diagnostic

Evaluation-only PPO pairing overrides were added to `scripts/evaluate_ml.py`
and tested without changing the default checkpoint behavior. The same frozen
PPO-451 checkpoint was evaluated with GA and Couto pairing using fresh seed
`900451`, 3,000 games per matchup, MC-64, alternated seats, raw game records,
and 5,000 paired-bootstrap samples. Artifacts are in:

```text
runs/ppo_pairing_diagnostic_seed900451/
```

| Frozen checkpoint pairing | Rule28-smart | MC | Minimum |
|---|---:|---:|---:|
| GA | 65.3% | 56.9% | 56.9% |
| Couto | 63.7% | 56.1% | 56.1% |

GA minus Couto was +1.60 percentage points against `rule28-smart` (paired 95%
interval -0.47 to 3.63) and +0.87 against MC (interval -1.17 to 3.00). Neither
difference was significant, and Couto did not improve either point estimate.
The MC rate variation from the prior seed illustrates why common-seed paired
comparisons, rather than cross-run point estimates, drive the decision.

Decision: do not train a fixed-Couto PPO. Preserve GA as the fixed-pairing
control. If pursuing another substantive PPO experiment, implement joint
learned pairing with phase-specific legal-action masking and initialize the
Stop/Continue representation from PPO-451; do not use a deeper fixed-pairing
network as the next experiment.

### Joint learned-pairing PPO implementation ready

The joint PPO pipeline was implemented on 2026-10-27. It uses the existing
86-value joint observation (68 domain values plus 18 pairing-candidate values),
six canonical pairing actions, and Continue/Stop actions 6--7. Phase-specific
legal masks are applied both when collecting actions and when recomputing PPO
log probabilities and entropy during every optimization epoch.

`--initial-ppo` expands PPO-451 into the joint model by copying its original
state-feature weights, hidden layers, value head, and Continue/Stop rows. The
new 18 input columns start at zero. A deterministic GA imitation stage trains
only those new input columns and pairing output rows 0--5, preserving all
transferred behavior on Stop/Continue decision states, before all parameters
are jointly optimized by PPO. Existing two-action PPO
checkpoint loading and fixed-pairing evaluation remain backward compatible;
joint checkpoints use learned pairing automatically during complete-game
evaluation.

A 20-episode seed-452 smoke run completed training, two validation passes,
checkpoint writing/loading, and a separate complete-game evaluator check:

```text
runs/ppo_joint_pairing_smoke_seed452/
```

The smoke used only 64 imitation examples, a random opponent, and tiny
evaluation settings. Its win rates are pipeline checks and are not experimental
evidence. The next experiment is the single predeclared seed-452 100,000-episode
joint run documented in `README.md`, retaining the 128x128 network, terminal-only
reward, 60% MC / 40% `rule28-smart` mixture, MC-16 training, and PPO-451
initialization. Do not use seeds `700450`, `800451`, or `900451` for its final
held-out comparison.

The first full seed-452 launch in `runs/ppo_joint_pairing_seed452/` stopped at
approximately 11,000 episodes because a rare seat-1 game ended during the
automated opponent's opening turn, leaving no learner observation for that
episode. This was a trainer bookkeeping defect, not an illegal policy action or
numerical PPO failure. The trainer now counts and replaces observationless
episodes; a deterministic regression test reproduces the exact opening-turn
condition. Before restart, imitation was also improved to train the new 18
pairing-feature columns while leaving PPO-451's decision-state behavior exact.
The corrected restart directory is `runs/ppo_joint_pairing_seed452_v2/`; the
partial original run is not experimental evidence.

### Next experiment: pairing-only PPO diagnostic

The joint run showed a large rise in bust rate and a collapse in win rate, so
the next run must not update the transferred Stop/Continue policy. The trainer
now supports `--freeze-decision-policy`: PPO updates can change only the 18 new
pairing input columns and pairing output rows 0--5. Gradients to the original
68 state columns, hidden layers after the first layer, value head, and
Continue/Stop rows 6--7 are excluded. A 20-episode smoke test and dedicated
gradient-mask regression test pass.

The predeclared next run is seed `453`, with the same opponent mix and PPO
settings as seed 452, in:

```text
runs/ppo_pairing_only_seed453/
```

Its purpose is diagnostic: determine whether learned pairing can add value while
preserving PPO-451. The result should be compared against PPO-451 under the
same fresh common evaluation seeds. Do not promote it based on its internal
500-game validation or its final checkpoint alone.

The seed-453 run was stopped early after approximately 50,000 episodes because
the diagnostic was consistently weak. It did not produce a final held-out
evaluation, so it is not formal comparative evidence. Its repeated 500-game
validation values were approximately 50--52% against `rule28-smart` and
41--46% against MC; the best recorded minimum was only 45.6%. Importantly,
its behavior remained near the transferred PPO-451 regime (`0.435--0.456`
busts/turn and `0.54--0.56` stops/turn), unlike the unfrozen seed-452 joint run
whose bust rate rose to about `0.55`. This isolates the likely failure to the
learned pairing branch rather than Stop/Continue destruction.

Decision: stop the learned-pairing PPO branch. Retain PPO-451 with fixed GA
pairing as the strongest PPO candidate. Do not spend another long training run
on this architecture. If a final confirmation is desired, evaluate
`runs/ppo_pairing_only_seed453/top_1.pt` against PPO-451 and DDQN on one fresh
common held-out seed; otherwise proceed with the retained PPO-451/DDQN
comparison and report learned pairing as a negative diagnostic result.

That common held-out evaluation was completed with seed `1000453`, 3,000 games
per matchup, MC-64, alternated seats, raw records, and 5,000 paired-bootstrap
samples. Artifacts are in:

```text
runs/ppo_pairing_only_common_eval_seed1000453/
```

| Model | Rule28-smart | MC | Minimum |
|---|---:|---:|---:|
| PPO-451 + GA | 64.6% | 55.4% | 55.4% |
| Pairing-only seed 453 | 50.4% | 44.5% | 44.5% |
| DDQN reference | 61.4% | 52.8% | 52.8% |

PPO-451 exceeded pairing-only seed 453 by 14.1 points against Rule28-smart
(paired 95% interval 11.7--16.5) and 10.9 points against MC (8.5--13.4).
Pairing-only seed 453 was also below DDQN by 11.0 points against Rule28-smart
(-13.5 to -8.6) and 8.3 points against MC (-10.8 to -5.9). All reported paired
intervals excluded zero. This confirms learned pairing is a negative result in
the current formulation, not merely an unlucky validation checkpoint.

Final experimental direction: stop RL training, retain PPO-451 + GA as the
strongest learned policy, retain DDQN as the reference baseline, and use the
joint and pairing-only runs as documented ablations showing that learned dice
pairing did not transfer under the current terminal-reward setup.

## Phase-A counterfactual PPO proposal

At the request to pursue a stronger PPO, the next controlled experiment is now
implemented. It uses fixed GA pairing, strategic 80-feature observations, PPO
transfer from `runs/ppo_mc_focused_seed451/top_1.pt`, and an auxiliary MC
counterfactual action-margin loss. Labels are collected only on every eighth
eligible decision, with MC-8 training labels and future-roll limit 16. The
terminal PPO objective remains active, so this is not direct MC imitation.

The predeclared run was seed `454` in:

```text
runs/ppo_counterfactual_seed454/
```

The implementation smoke test completed, and the full regression suite passes
(100 tests). Promotion gates are: minimum validation above 52% by episode
30,000; then fresh common held-out evaluation against MC, Ridge, and
Rule28-smart. If the early gate fails, stop rather than tuning around it.

### Phase-A screening result: seed 454

The reduced-cost screening run completed 30,000 episodes in
`runs/ppo_counterfactual_screen_seed454/` using MC-2 labels every 64 decisions.
It passed the mechanical early gate only narrowly: the episode-30,000
validation was 53.7% against `rule28-smart` and 53.3% against MC (300 games
per opponent). Its best intermediate checkpoint was episode 10,059 at 57.0%
and 52.0%, but later training regressed.

The 1,000-game final evaluation of the selected episode-30,000 checkpoint was
53.6% against `rule28-smart` and 49.0% against MC. This is well below the
retained PPO-451 result of 64.6% and 55.4% on the prior common held-out
comparison, though the seeds and game counts are not identical. The behavior
also drifted from the transferred policy: greedy stop rate fell to about 9.3%
and busts per turn reached about 0.48. The counterfactual loss therefore did
not preserve PPO-451's robust behavior in this run.

Decision: do not expand this Phase-A run to 100,000 episodes or increase label
density. The result is a negative screening result, not evidence that the
strategic representation is useless; the reduced MC-2/every-64 labels and
terminal PPO updates were not strong enough to produce a robust improvement.
Retain PPO-451 + GA as the promoted PPO candidate unless a separately
predeclared design changes the training objective or freezes the transferred
policy more explicitly.

The directory `runs/ppo_mc_initialized_seed450/` contains only a configuration
and an empty validation file from an incomplete launch. It is not experimental
evidence.

Cleanup: all non-promoted Phase-A seed-454 artifacts were removed after this
record was written: `ppo_counterfactual_screen_seed454`,
`ppo_counterfactual_seed454`, and `ppo_counterfactual_smoke_seed454`. The
retained PPO-451, DDQN reference, and prior held-out comparison artifacts were
not modified.

### Final common evaluation including Ridge

The final comparison was completed with 3,000 games per matchup, evaluation
seed `1000454`, MC-64, alternating seats, and 5,000 paired-bootstrap samples.
Artifacts are in `runs/final_common_eval_seed1000454/`.

| Model | Rule28-smart | MC | Minimum |
|---|---:|---:|---:|
| PPO-451 + GA | **64.6%** | **55.4%** | **55.4%** |
| DDQN reference | 61.4% | 52.7% | 52.7% |
| Ridge-domain | 60.1% | 51.1% | 51.1% |

PPO-451 exceeded DDQN by 3.2 points against Rule28-smart (paired 95% interval
1.2--5.3) and 2.7 points against MC (0.6--5.0). It exceeded Ridge by 4.5
points against Rule28-smart (2.2--6.7) and 4.3 points against MC (1.9--6.6).
All four intervals excluded zero. DDQN versus Ridge was not statistically
resolved for either opponent. This confirms PPO-451 is the strongest of the
three evaluated learned/classical models on this common held-out seed, although
the result is not evidence of a 55--60% win rate against every opponent and
does not by itself establish multi-seed generalization.

Decision: promote PPO-451 + GA as the final PPO candidate; retain DDQN and
Ridge as comparison baselines. Do not continue the failed counterfactual PPO
branch. Use `runs/final_common_eval_seed1000454/` and the earlier common
evaluation as report evidence, while describing confidence intervals as
paired-seed descriptive inference and not independent-game intervals.

### Four-model direct head-to-head

The direct four-model comparison used 3,000 games per pair, seed `60000643`,
MC-64 where applicable, and 5,000 paired-bootstrap samples. Results are in
`runs/four_model_round_robin_seed60000643/`.

| Matchup | First-model win rate | Paired 95% interval |
|---|---:|---:|
| PPO-451 vs DDQN | 53.9% | 52.3--55.6% |
| PPO-451 vs Ridge | 53.9% | 52.2--55.6% |
| DDQN vs Ridge | 49.0% | 47.3--50.8% |
| PPO-451 vs Random Forest | 77.5% | 75.9--79.0% |
| DDQN vs Random Forest | 75.5% | 73.9--77.0% |
| Ridge vs Random Forest | 75.5% | 73.9--77.0% |

The model matrix confirms PPO-451 is the strongest direct competitor. Ridge and
DDQN are approximately tied, while `random-forest_domain` is decisively weaker
than every retained model despite its supervised validation metrics. This is a
useful reminder that offline classification metrics do not establish complete
game strength. Random Forest should not be treated as one of the two strongest
classical candidates; Ridge is the only currently supported classical model
selection.

### Four-model evaluation against external agents

The external-opponent evaluation used 3,000 games per matchup, seed `60000644`,
MC-64, and 5,000 paired-bootstrap samples. Artifacts are in
`runs/four_models_vs_external_seed60000644/`.

| Model | Rule28-smart | MC | Glenn--Aloi | Couto-GA-stop | Minimum |
|---|---:|---:|---:|---:|---:|
| PPO-451 | **64.1%** | **55.3%** | 80.9% | **65.8%** | **55.3%** |
| DDQN | 60.6% | 52.6% | 79.2% | 63.2% | 52.6% |
| Ridge | 59.1% | 52.1% | **81.7%** | 63.7% | 52.1% |
| Random Forest | 34.2% | 28.4% | 69.4% | 38.7% | 28.4% |

PPO-451 has the strongest minimum and leads DDQN on all four external
opponents. Ridge is slightly better against Glenn--Aloi, demonstrating
opponent-dependent rankings, but trails PPO on the other three. Random Forest
is decisively weak despite acceptable offline classification metrics. It should
be removed from the final model set; retain PPO-451, DDQN, and Ridge.

## Conservative self-play fine-tuning result

The completed run `ddqn_finetune_conservative_mix_seed447_v2` was initialized
from `runs/ddqn_mc_counterfactual_seed447/best.pt` and trained for 5,000
additional episodes. Its intended opponent distribution was 20% frozen
self-play, 35% `rule28-smart`, 35% `mc`, and 10% `ridge-domain`. The replay
buffer was reinitialized, training used one-rollout MC anchors, and the
training safety limit was 300 turns.

The run completed, but the model regressed badly. In 3,000 held-out games per
opponent, its win rates were 34.3% against `rule28-smart`, 29.1% against
`mc`, and 26.7% against Ridge. The minimum win rate was 26.7%. Its greedy
Stop rate increased to 21.3% while busts per turn fell to 0.220, indicating an
overly conservative policy that stopped too often to make sufficient progress.

Decision: do not promote this checkpoint. The run artifacts were removed after
recording these results; the experiment is retained here as a negative result.

## Targeted fine-tuning result

The completed run `ddqn_targeted_finetune_seed449` was initialized from the
retained reference checkpoint and fine-tuned for 1,000 episodes with a fresh
replay buffer, learning rate `1e-5`, fixed epsilon `0.01`, and a 45%/45% mix
of `rule28-smart`/`mc` opponents plus 10% random opponents.

Its 3,000-game held-out results were:

| Model | Rule28-smart | MC | Minimum |
|---|---:|---:|---:|
| Reference robustness result | 60.9% | 53.9% | 53.9% |
| Targeted fine-tune | 55.3% | 47.3% | 47.3% |

The candidate became more conservative: busts per turn were approximately
`0.292` against Rule28-smart versus `0.439` for the reference, while its
stopping rate increased to approximately 71%. The short validation peak of
53.2% at episode 500 did not transfer to held-out evaluation.

Decision: do not promote. The run was removed during cleanup after recording
these results; retain the original DDQN reference.

## Aborted Ridge-opponent experiment

Run:

```text
runs/ddqn_ridge_robust_seed447/
runs/ddqn_ridge_robust_seed447.log
```

Changes relative to the weight-0.05 reference were limited to the training
opponent distribution: 20% `ridge-domain`, with the original opponents scaled
to the remaining 80%. The Ridge model was
`runs/classical_ml_seed541/models/ridge_domain.joblib`.

The run was stopped at approximately 53,401/100,000 episodes. It had no
formal held-out head-to-head evaluation. Greedy validation was poor and
unstable:

| Episode | Rule28-smart | MC | Minimum | Busts/turn | Greedy stop |
|---:|---:|---:|---:|---:|---:|
| 5,001 | 22.3% | 23.4% | 22.3% | 0.743 | 3.8% |
| 30,001 | 51.0% | 48.9% | 48.9% | 0.521 | 8.4% |
| 50,000 | 52.2% | 45.9% | 45.9% | 0.524 | 8.3% |

Interpretation: the 20% Ridge mixture produced an overly aggressive policy
(very low Stop rate and high bust rate) and did not improve the target
opponents. Do not use its `best.pt` as the final model; retain it only as an
aborted-experiment artifact.

## Reference results

## Robustness matrix evaluation

On 2026-10-26, four available checkpoints were evaluated against the same
five opponents using 3,000 games per matchup, seed `600447`, alternating
seats, and 5,000 paired-bootstrap samples. Raw games and summaries are in:

```text
runs/dqn_robustness_parallel_seed600447/
```

The results were:

| Checkpoint | Rule28-smart | MC | Random | Rule28 | Ridge | Minimum |
|---|---:|---:|---:|---:|---:|---:|
| MC-counterfactual reference | 60.9% | 53.9% | 96.2% | 77.2% | 50.3% | 50.3% |
| Anchored self-play | 60.8% | 49.8% | 96.5% | 76.6% | 49.4% | 49.4% |
| Robust seed 441 | 60.3% | 51.0% | 95.1% | 75.3% | 48.4% | 48.4% |
| Robust seed 445 | 60.4% | 51.6% | 97.4% | 74.7% | 48.8% | 48.8% |

The reference checkpoint remains the strongest of the evaluated models. The
anchored self-play checkpoint does not improve robustness: it is nearly tied
against `rule28-smart`, but loses 4.1 percentage points against MC and 0.9
points against Ridge relative to the reference. The two available robust
seeds show the same pattern, with Ridge win rates below 50%.

Decision: do not promote any of these alternatives. This evaluation supports
testing policy-preserving fine-tuning rather than more self-play training.

## Reference-policy regularized fine-tuning E1

The first policy-preserving fine-tuning attempt used the retained reference
checkpoint with a frozen-reference KL penalty (`lambda=0.1`). It trained for
1,000 additional episodes against 40% `rule28-smart`, 40% `mc`, and 20%
`ridge-domain`, with learning rate `3e-5` and epsilon fixed at `0.03`.

The run artifacts were removed during cleanup after recording these results.

The run completed, but it regressed. Its final 100-game validation was 52.0%
against `rule28-smart`, 54.0% against `mc`, and 45.0% against Ridge. The
3,000-game held-out selection score was only 40.9%. The policy became more
conservative than the reference: busts per turn fell to 0.283, stops per turn
rose to 0.717, and greedy Stop rate reached 17.2%.

Decision: do not promote E1 and do not immediately run larger values of the
same regularizer. A KL penalty on all Q outputs did not prevent policy drift
under a newly initialized replay buffer; future fine-tuning should first use
an explicitly preserved/replayed reference transition distribution or a much
shorter update budget.

## Cleanup decision

On 2026-10-26, obsolete model artifacts were removed after their results were
recorded. The deleted items were the complete E1 regularized fine-tuning run,
the anchored self-play training run, the two non-promoted robust-seed
checkpoints, and the older partial robustness output superseded by the
parallel robustness matrix. The retained evidence is the reference run,
`runs/dqn_robustness_parallel_seed600447/` (raw games and summaries),
`runs/ddqn_robust_5seeds/multi_seed_summary.json`, and the classical-ML
dataset/models needed for baseline evaluation.

## Next experiment proposal: reference-replay fine-tuning

F0 implementation is complete and the replay round-trip smoke test passed on
2026-10-26. F1 collected 60,000 frozen-policy 5-step transitions and completed
the short fine-tuning run; its artifacts were removed after recording the
negative result below.

Do not start another long self-play or KL-only run. The next experiment should
test whether replaying the reference policy's transition distribution prevents
the drift seen in E1 while exposing the learner to Ridge-specific states.

### F0: implementation and short smoke test

Add a replay export/import path for fixed-length transitions, including state,
action, reward, next state, terminal flag, n-step count, and counterfactual
fields. Generate a frozen reference replay set with the retained checkpoint
using deterministic seeds and the existing 40% `rule28-smart`, 40% `mc`, 20%
`ridge-domain` mixture. Verify that importing the replay produces identical
sampled tensors under a fixed replay RNG seed.

### F1: short policy-preserving update

Start from `runs/ddqn_mc_counterfactual_seed447/best.pt`, prefill replay with
the frozen reference transitions, then collect new transitions with the same
opponent mixture. Use a 75% reference / 25% new-transition sampling mixture,
learning rate `3e-5`, epsilon `0.03`, and only 5,000 gradient updates. Keep the
existing DDQN, GA pairing, domain features, n-step target, and
counterfactual-loss settings unchanged. Do not use self-play and do not add a
new KL penalty in the first test; this isolates replay preservation.

Validate at 1,000-update intervals with at least 500 games per opponent. Stop
and reject a candidate if either `rule28-smart` or `mc` falls more than 3
percentage points below the reference, or if Ridge does not improve by the
final checkpoint. Only a surviving candidate receives the 3,000-game held-out
evaluation against `rule28-smart`, `mc`, Ridge, and Random using a fresh seed.
Promotion requires improvement against Ridge without reducing either primary
opponent by more than 3 points. If F1 fails, stop fine-tuning and report the
retained DDQN as the final model rather than escalating update count or KL
weight.

### F1 result

F1 used a 75% reference-replay / 25% new-transition mixture, learning rate
`3e-5`, epsilon `0.03`, and 500 episodes (approximately 4,000 new gradient
updates after the historical checkpoint counter). The 3,000-game held-out
results were:

| Model | Rule28-smart | MC | Ridge | Minimum |
|---|---:|---:|---:|---:|
| Reference robustness result | 60.9% | 53.9% | 50.3% | 50.3% |
| F1 reference replay | 58.6% | 50.4% | 46.7% | 46.7% |

F1 failed the promotion gate: all three matchups regressed, including Ridge.
Its validation at episode 500 was 57.0% / 51.0% / 45.8% for Rule28-smart / MC /
Ridge, with busts per turn `0.336` and greedy Stop rate `14.1%`. The F1 model,
replay file, and log were deleted after these results were recorded. No further
fine-tuning variants are planned; retain the original DDQN reference.

## Reward-shaping ablation

The terminal-only condition is already represented by the retained reference
(`reward_shaping_scale=0.0`). On 2026-10-26, the first new reward ablation was
started from scratch with progress-potential shaping at scale `0.25`, seed
`448`, and all other DDQN/domain-feature/GA-pairing settings held constant.
The run is:

```text
runs/ddqn_reward_progress025_seed448/
```

Validation includes `rule28-smart`, `mc`, and `ridge-domain`. The shaping run
will only be promoted after a fresh held-out evaluation beats the terminal-only
reference without sacrificing either primary opponent.

The progress-shaping run was intentionally stopped at approximately episode
17,500 before its first formal validation checkpoint because the single-process
trainer was too slow (about 350 environment steps/second and an estimated
additional 74 minutes). Its partial checkpoints and log were deleted during
cleanup; this remains an incomplete run, not experimental evidence.

Formal paired evaluation of the weight-0.1 counterfactual model gave 61.58%
against `rule28-smart` and 53.62% against `mc`, with paired improvements over
the seed-441 control but substantially higher bust rates. The later direct
Ridge matchups showed strategy-specific weakness: Ridge won 70.8% against the
weight-0.05 candidate but only 50.5% against the seed-441 control. This is
evidence of policy non-transitivity, not evidence that the counterfactual
model is generally stronger.

## 2026-10-26 cleanup

The following non-promoted artifacts were deleted after recording their status:

- `runs/ddqn_reward_progress025_seed448/` and its log: incomplete reward-shaping run.
- `runs/double-dqn_raw_deep_joint_seed448/`: 2,000-episode smoke model with
  5.2% and 6.2% win rates against `rule28-smart` and `mc`.
- `runs/ddqn_reference_replay_f1_seed447.log`: log for the already-rejected F1
  reference-replay experiment.

The non-promoted `runs/ddqn_reward_progress025_seed448_v2/` artifacts were
also removed during cleanup. No pending fine-tuning run should be treated as
a promotion candidate; retain the original DDQN reference.

## Historical frozen-snapshot self-play result

The pure self-play artifacts were removed during experiment cleanup after
they failed to improve the retained DDQN. Self-play win rate alone is not
evidence of improvement because the learner and its snapshots are not
independent external opponents.

### Self-play result

The run completed all 100,000 episodes using frozen learner snapshots every
5,000 episodes, with a 50% newest-snapshot selection probability and a
20-snapshot history pool. It was evaluated against `rule28-smart` and `mc`,
but not yet against Ridge.

The best 1,000-game validation occurred at episode 60,001:

| Opponent | Validation win rate |
|---|---:|
| `rule28-smart` | 55.4% |
| `mc` | 50.3% |

The selected checkpoint achieved 54.96% against `rule28-smart` and 48.72%
against `mc` in 5,000 held-out games, for a minimum score of 48.72%. The final
episode-100,000 model was worse at 41.78% and 36.40%, showing late-training
regression.

Compared with the retained counterfactual DDQN on the same held-out seed
(60.48% and 55.02%), self-play was lower by 5.52 and 6.30 percentage points.
The selected checkpoint also had high bust rates (0.455 and 0.443) and a
greedy Stop rate of only 10.6%.

Decision: do not promote the self-play model. The snapshot mechanism trains
successfully, but this pure self-play setup did not outperform the existing
DDQN and showed checkpoint/strategy instability.

### Self-play head-to-head follow-up

The selected self-play checkpoint was compared directly with Ridge and the
seed-441 control using 5,000 paired games:

| Matchup | Model A win rate | Paired bootstrap 95% CI |
|---|---:|---:|
| Ridge vs self-play | 53.36% | 51.98%--54.72% |
| Ridge vs control | 50.46% | 49.10%--51.84% |
| Self-play vs control | 47.64% | 46.32%--49.02% |

Self-play had substantially higher bust rates than both opponents (0.448 vs
Ridge and 0.453 vs control), compared with approximately 0.34 for the other
agent in each matchup. The direct comparison confirms that self-play lost to
the control DDQN and was also beaten by Ridge. Ridge and control were
statistically indistinguishable in this matchup.

### Supervised DNN on the classical counterfactual dataset

To test whether extra neural-network capacity could improve the classical baseline, a supervised DNN was trained on exactly the same 10,000-state dataset and grouped split used by Ridge. The target remained the rollout-derived `continue_margin`; the DNN used standardized 68-dimensional domain features and a `68-128-128-1` MLP. The checkpoint and training report are in `runs/supervised_dnn_seed641/`.

The validation selection metrics were 68.2% action accuracy and 0.0181 mean rollout regret. However, the complete-game held-out evaluation used the same 3,000 games per matchup, MC-64, and paired-bootstrap protocol for both models:

| Model | Rule28-smart | MC | Minimum |
|---|---:|---:|---:|
| Supervised DNN | 42.7% | 36.2% | 36.2% |
| Ridge-domain | 60.6% | 53.2% | 53.2% |

Artifacts are in `runs/supervised_dnn_vs_baseline_seed20260641/`. The DNN did not improve the classical baseline and is not promoted. This negative result is important: supervised state/action accuracy and rollout regret do not guarantee that a policy will perform well over complete games. The DNN may be overfitting the noisy counterfactual margin or may be sensitive to distribution shift from the collected Rule28-smart states; further DNN tuning is not justified without a new predeclared experiment design.

### Multi-policy state-distribution DNN

A second supervised DNN experiment collected 10,000 states from five policies: Rule28-smart, MC, Ridge, PPO-451, and DDQN. These policies supplied only the state distribution; all labels were generated identically by forced Stop/Continue counterfactual rollouts followed by the Rule28-smart rollout policy. The dataset is `runs/multi_policy_counterfactual_dataset_seed20261028.csv`, and the DNN is in `runs/supervised_dnn_multi_policy_seed641/`.

The common held-out evaluation used 3,000 games per matchup, seed `20261029`, MC-64, and 5,000 paired-bootstrap samples:

| Model | Rule28-smart | MC | Minimum |
|---|---:|---:|---:|
| Multi-policy DNN | 32.3% | 28.7% | 28.7% |
| Ridge-domain | 59.5% | 53.2% | 53.2% |
| PPO-451 | 65.0% | 55.2% | 55.2% |
| DDQN | 61.1% | 52.3% | 52.3% |

The multi-policy DNN is not promoted. Its validation action accuracy was only 59.1%, lower than the single-source DNN, and its complete-game policy was substantially weaker than every retained model. This suggests that simply broadening the state distribution while retaining a Rule28-based counterfactual labeler does not produce a stronger policy; the mixed state distribution may make the approximate labels less coherent for a single global regressor. No further DNN tuning is planned without a new objective or a predeclared state-conditioned labeling design.

### Ridge trained on the multi-policy state distribution

As a controlled follow-up to the multi-policy DNN, Ridge was trained on the same 10,000 multi-policy states and the same common counterfactual labels. The resulting artifact is `runs/ridge_multi_policy_seed641/ridge_domain.joblib`; evaluation artifacts are in `runs/ridge_multi_policy_eval_seed20261030/`.

| Model | Rule28-smart | MC | Minimum |
|---|---:|---:|---:|
| Ridge multi-policy | 62.7% | 57.4% | 57.4% |
| Original Ridge | 59.4% | 53.2% | 53.2% |
| PPO-451 | 65.0% | 55.2% | 55.2% |
| DDQN | 61.1% | 52.3% | 52.3% |

Relative to original Ridge, the multi-policy Ridge improvement was +3.3 points against Rule28-smart (paired 95% interval +1.0--+5.6) and +4.2 points against MC (+2.0--+6.7); both intervals excluded zero. It exceeded DDQN by 5.1 points against MC (+2.9--+7.2), while its +1.6-point Rule28-smart advantage over DDQN was not resolved. PPO remained higher against Rule28-smart by 2.3 points (interval -4.4-- -0.3), while the +2.2-point PPO advantage against MC was not resolved.

This result supports the hypothesis that broadening the state distribution can help a simple model, but the benefit is model-dependent: multi-policy Ridge improved substantially whereas the multi-policy DNN collapsed. The multi-policy Ridge is a promising additional robustness candidate, but the original final model decision remains PPO-451 as the primary model; this single new evaluation does not establish multi-seed generalization.

### Final two-seed retraining check

The final confirmation run retrained each retained model with two additional seeds using the same configurations: PPO seeds 460/461, DDQN seeds 448/449, and multi-policy Ridge seeds 642/643. The common evaluation used 3,000 games per matchup, seed `20261101`, MC-64, and 5,000 paired-bootstrap samples. Head-to-head evaluation used seed `20261102`.

| Model family | Rule28-smart mean | MC mean | External mean | New-seed minimum mean |
|---|---:|---:|---:|---:|
| PPO (460/461) | 64.25% | 55.35% | 74.6% | 55.35% |
| DDQN (448/449) | 59.50% | 50.40% | 71.0% | 50.40% |
| Ridge (642/643) | 63.25% | 55.55% | 74.9% | 55.55% |

The two PPO seeds were consistent. DDQN showed larger seed sensitivity, with seed 448 clearly stronger than 449. Ridge was stable across its two new seeds and remained close to PPO in direct matchups. The previously canonical Ridge checkpoint remains the strongest single Ridge artifact against MC (57.4% in the final retraining evaluation), but it is not used to claim a multi-seed average. No further model training is planned; these results are for final robustness analysis and reporting.

### Direct comparison with selected rule, Monte Carlo, and paper agents

On 2026-10-27, the retained DDQN, PPO, and Ridge checkpoints were evaluated
against the selected comparison agents: `rule28-smart`, `mc-strong`,
`glenn-aloi`, `glenn2009_constant`, `glenn2010_linear`, and `couto`. The four
paper agents were also evaluated directly against `mc-strong` and
`rule28-smart`. Each matchup used 3,000 games, alternating seats, 64 MC
rollouts, and a 2,000-turn limit. Matchup-specific seed offsets were used;
results were printed without saving logs or raw game records. No confidence
intervals were computed.

| Model | rule28-smart | mc-strong | glenn-aloi | glenn2009_constant | glenn2010_linear | couto |
|---|---:|---:|---:|---:|---:|---:|
| DDQN | 61.2% | 50.5% | 79.0% | 66.2% | 61.3% | 60.2% |
| PPO | **63.5%** | 53.7% | **81.3%** | **67.2%** | 67.1% | **64.6%** |
| Ridge | 62.9% | **54.4%** | **81.8%** | 66.0% | **68.9%** | 64.1% |

| Paper agent | vs mc-strong | vs rule28-smart |
|---|---:|---:|
| glenn-aloi | 20.2% | 22.7% |
| glenn2009_constant | 36.9% | 41.0% |
| glenn2010_linear | 36.2% | 41.2% |
| couto | 41.1% | 45.7% |

PPO was strongest overall among the three retained project models. Ridge was
strongest against `mc-strong` and `glenn2010_linear`, while DDQN was nearly
even with `mc-strong`. `mc-strong` substantially outperformed all four paper
agents in the direct comparison. These point estimates are descriptive and do
not establish statistical significance.

### Selected paper agents against Random

As an additional calibration comparison on 2026-10-27, the four selected
paper agents were evaluated against `random`. Each matchup used 3,000 games,
alternating seats, a 2,000-turn limit, and no saved logs or raw game records.
There were no draws.

| Paper agent | Win rate vs Random | Wins |
|---|---:|---:|
| `glenn-aloi` | 78.0% | 2,341 / 3,000 |
| `glenn2009_constant` | 95.4% | 2,862 / 3,000 |
| `glenn2010_linear` | 94.2% | 2,827 / 3,000 |
| `couto` | **96.1%** | 2,882 / 3,000 |

`couto` had the highest win rate against `random`, followed closely by the
2009 Constant and 2010 Linear agents. `glenn-aloi` was substantially weaker
in this calibration matchup.

### Post-merge paper-agent and retained-model rerun

After PR #1 was merged as `0d53b965`, the README comparison was rerun because
the PR changes published-policy behavior and legal-move ordering. The retained
model run used PPO seed 460 (`top_1.pt`), DDQN seed 448 (`best.pt`), and Ridge
seed 642 (`ridge_domain.joblib`). It used 3,000 games per matchup, seed
`20261103`, alternating seats, a 2,000-turn limit, MC-64, four workers, and
5,000 paired-bootstrap samples. Raw game records, the matrix, and the summary
are archived in `docs/experiments/post_merge_20261103/`.

| Model | rule28-smart | mc-strong | glenn-aloi | glenn2009_constant | glenn2010_linear | couto |
|---|---:|---:|---:|---:|---:|---:|
| PPO-460 | 64.8% | 53.4% | 68.1% | 69.2% | 66.4% | 67.6% |
| DDQN-448 | 61.0% | 51.3% | 64.1% | 64.8% | 60.3% | 66.1% |
| Ridge-642 | 63.4% | 54.0% | 66.9% | 68.0% | 65.7% | 68.3% |

The paper agents' own rates in the same matrix were 43.0%, 41.5%, 43.4%, and
43.1% against `rule28-smart`, and 39.4%, 37.1%, 37.7%, and 37.1% against
`mc-strong`, in the order Glenn–Aloi, Glenn 2009, Glenn 2010, and Couto.
A separate 3,000-game Random calibration with seed `20261027` gave 95.6%,
95.3%, 97.5%, and 95.3%, respectively. These results supersede the earlier
pre-merge point estimates in this document and in the README. The old run did
not save raw records, so exact reconstruction of its seed offsets is not
possible; the new run is the reproducible reference.

### Carter public checkpoint adaptation

The public Carter repository publishes `Code/dqn_multi_opp.keras` at
`JCameronCarter/HonorsThesis`. This checkpoint was loaded as the reported
`51 -> 64 -> 64 -> 2` network, with Carter's heuristic dice-pair selector and
the action convention `0=stop`, `1=continue`. Because TensorFlow was not
available in the evaluation environment, the weights inside the Keras archive
were read directly and evaluated with an equivalent PyTorch inference network;
the architecture and weights were not retrained or modified.

Using the current simulator's 51-feature adapter, seed `20261104`, alternating
seats, 3,000 games per matchup, and a 2,000-turn limit, Carter's adapted
checkpoint produced:

| Opponent | Carter wins | Carter win rate |
|---|---:|---:|
| Random | 594 / 3,000 | 19.8% |
| rule28-smart | 51 / 3,000 | 1.7% |
| mc-strong | 52 / 3,000 | 1.7% |
| PPO-460 | 34 / 3,000 | 1.1% |
| DDQN-448 | 44 / 3,000 | 1.5% |
| Ridge-642 | 53 / 3,000 | 1.8% |
| glenn-aloi | 22 / 3,000 | 0.7% |
| glenn2009_constant | 19 / 3,000 | 0.6% |
| glenn2010_linear | 34 / 3,000 | 1.1% |
| couto | 46 / 3,000 | 1.5% |

These results are a **checkpoint and feature/simulator adaptation**, not a
reproduction of Carter's original game environment or training evaluation.
The 19.8% result against Random is evidence that the feature and rule
translation is not behaviorally equivalent; therefore these numbers should
not be used to claim that Carter's original agent performs this poorly under
its own environment. The public checkpoint provenance is retained here so
the comparison is transparent.

The same Random calibration was extended to the Carter adaptation using seed
`20261027`, 3,000 games, alternating seats, and a 2,000-turn limit. Carter won
749 games (24.97%, reported as 25.0% in the README). The saved result is
`docs/experiments/post_merge_20261103/random_vs_paper/04_carter_vs_random.json`.
This remains a simulator/feature adaptation, not a reproduction of Carter's
original-environment result.
