# Final evidence package

This directory contains the compact evidence package for the report and video.
The CSV is the source for the three SVG figures and records the provenance of
each number. The underlying experiment-status document remains the source of
truth for full methodology and caveats.

The accompanying [`CONSISTENCY_AUDIT.md`](CONSISTENCY_AUDIT.md) records why
the post-merge comparison and the final common-held-out model comparison are
kept as separate evidence sets.

## Promoted conclusion

`PPO-451 + GA` is the strongest retained model on the common held-out
evaluation. It reached 64.6% against `rule28-smart` and 55.4% against `MC`,
exceeding the retained DDQN by 3.2 and 2.7 percentage points. The reported
paired intervals for those differences excluded zero on that fixed evaluation.
This is evidence for the selected checkpoint, not a claim of universal
multi-seed dominance.

## Main-model table

| Model | Rule28-smart | MC | Evaluation |
|---|---:|---:|---|
| PPO-451 + GA | **64.6%** | **55.4%** | 3,000 games; seed 1000454 |
| DDQN reference | 61.4% | 52.7% | 3,000 games; seed 1000454 |
| Ridge-domain | 60.1% | 51.1% | 3,000 games; seed 1000454 |

PPO minus DDQN was +3.2 percentage points (paired 95% interval 1.2--5.3)
against Rule28-smart and +2.7 points (0.6--5.0) against MC. PPO minus Ridge
was +4.5 points (2.2--6.7) and +4.3 points (1.9--6.6), respectively.

## Negative-result table

| Model | Rule28-smart | MC | Interpretation |
|---|---:|---:|---|
| PPO-451 + GA | 64.6% | 55.4% | retained control |
| Pairing-only PPO | 50.4% | 44.5% | learned pairing failed to transfer |

The pairing-only model was lower by 14.1 points against Rule28-smart and 10.9
points against MC; both paired intervals excluded zero.

## Search-budget table

| Rollout budget | Flat Monte Carlo | Root-only UCB1 |
|---:|---:|---:|
| 16 | 0.3% | 0.8% |
| 64 | 3.3% | 4.3% |
| 256 | 10.3% | 11.5% |

These are means over two 500-game replicates per budget, not significance
claims.

## Figures

1. [`primary_model_advantage.svg`](primary_model_advantage.svg): PPO's paired
   advantage over DDQN and Ridge on the common held-out evaluation.
2. [`learned_pairing_negative.svg`](learned_pairing_negative.svg): the learned
   pairing diagnostic, where pairing-only PPO lost to PPO-451 + GA.
3. [`vmerckle_budget_sweep.svg`](vmerckle_budget_sweep.svg): descriptive win
   rates as rollout budget increased.

The figures are intentionally SVG and require no plotting package to rebuild:

```bash
python3 scripts/generate_final_figures.py
```

## Evidence boundaries

- Main-model numbers use 3,000 games per matchup, alternating seats, and
  paired bootstrap inference; the intervals shown are paired differences, not
  independent-game confidence intervals.
- The learned-pairing result is a formal negative result, not an abandoned
  screening run.
- The vmerckle sweep uses two 500-game replicates per budget in the archived
  experiment, so its displayed means are descriptive.
- No new training or unsupported prior-work reproduction is implied by this
  package.
