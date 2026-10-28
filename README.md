# Can't Stop — Learning Risk-Aware Game Strategy

> **Can reinforcement learning learn when to push forward—and when to stop—better than hand-designed rules?**

This project studies that question in *Can't Stop*, a board game whose central decision is a compact but difficult form of risk management: every additional roll can create progress, but can also erase the progress made during the current turn. We built and trained our own **Ridge**, **Double DQN (DDQN)**, and **PPO** agents, implemented a stronger **`mc-strong`** Monte Carlo opponent, and created a reproducible evaluation pipeline to compare them under identical dice, seats, and held-out games.

The project is important beyond one board game: it is a controlled testbed for sequential decision-making under uncertainty, where short-term reward, catastrophic failure, opponent behavior, and non-transitive policies interact.

## Headline result: our agents versus prior work

Our models are not merely classifiers that imitate labels: they play complete games against **traditional baselines and agents from previous work**. In a common direct comparison of **3,000 games per matchup**, our PPO, DDQN, and Ridge agents beat the published Glenn–Aloi, Glenn 2009, Glenn 2010, and Couto strategies in several matchups. The stronger `mc-strong` opponent also exposes a useful research result: it beats all four prior heuristic agents, while our learned agents remain competitive against it.

The final two-seed retraining check showed that PPO and Ridge remained competitive while DDQN was more seed-sensitive. These are fresh complete-game evaluations—not training accuracy. The full experiment record, caveats, and raw-artifact locations are documented in [`EXPERIMENT_STATUS.md`](cant_stop_baselines/docs/EXPERIMENT_STATUS.md).

## Why this problem matters

*Can't Stop* makes risk-sensitive control visible and measurable:

- **Progress versus ruin:** continuing can advance multiple columns, but a bust loses all unbanked turn progress.
- **Delayed consequences:** the best action depends on board position, opponent threats, and how close the player is to completing columns.
- **A meaningful baseline gap:** simple rules can be strong, so an ML model must demonstrate improvement in complete games—not merely fit labels.
- **A clean scientific setting:** the simulator gives us exact rules, deterministic seeds, controlled opponents, and repeatable ablations.

## Evidence: our agents versus published agents

The following direct comparison used **3,000 games per matchup**, alternating seats and 64 Monte Carlo rollouts where applicable. These rows are our trained models; “win rate” means the row agent's win rate against the column opponent.

| Our agent | vs Rule-of-28-smart | vs `mc-strong` | vs Glenn–Aloi | vs Glenn 2009 | vs Glenn 2010 | vs Couto |
|---|---:|---:|---:|---:|---:|---:|
| **PPO** | **63.5%** | 53.7% | **81.3%** | **67.2%** | 67.1% | **64.6%** |
| **DDQN** | 61.2% | 50.5% | 79.0% | 66.2% | 61.3% | 60.2% |
| **Ridge** | 62.9% | **54.4%** | **81.8%** | 66.0% | **68.9%** | 64.1% |

For the reverse comparison, the prior agents' own win rates were much lower:

| Prior agent | vs `rule28-smart` | vs `mc-strong` |
|---|---:|---:|
| Glenn–Aloi [2] | 22.7% | 20.2% |
| Glenn 2009 Constant [2] | 41.0% | 36.9% |
| Glenn 2010 Linear [2] | 41.2% | 36.2% |
| Couto / Sketch-SA(O) [3] | 45.7% | 41.1% |

The `mc-strong` implementation therefore beats all four prior heuristic agents in this direct comparison, while our PPO, DDQN, and Ridge policies remain competitive against it.

These results show the main contribution clearly: our agents improve on the traditional baseline and defeat several previous agents, while no single policy dominates every opponent. The ranking is opponent-dependent, which is why we report a matrix rather than one cherry-picked score.

```mermaid
xychart-beta
    title "Our agents versus prior agents: Rule-of-28-smart"
    x-axis [PPO, DDQN, Ridge, GA, G09, G10, Couto]
    y-axis "Win rate (%)" 0 --> 90
    bar [63.5, 61.2, 62.9, 22.7, 41.0, 41.2, 45.7]
```

```mermaid
xychart-beta
    title "Our agents versus prior agents: mc-strong"
    x-axis [PPO, DDQN, Ridge, GA, G09, G10, Couto]
    y-axis "Win rate (%)" 0 --> 90
    bar [53.7, 50.5, 54.4, 20.2, 36.9, 36.2, 41.1]
```

These direct-comparison point estimates do not establish statistical significance because that run did not save bootstrap intervals. Formal common-seed evaluations with paired bootstrap inference are recorded in the experiment status document. Policy rankings are opponent-dependent, and the evaluation protocol uses paired seat-swapped games; we do not treat all games as independent samples.

## What we built

```mermaid
flowchart LR
    A[Exact Can't Stop rules] --> B[Deterministic seeded games]
    B --> C[Baseline agents]
    B --> D[Training transitions]
    D --> E[PPO / DDQN policies]
    C --> F[Held-out head-to-head evaluation]
    E --> F
    F --> G[Win rate, bust rate, paired bootstrap]
```

- **Rules engine:** immutable agent-facing state, legal move generation, bust handling, banked progress, column closure, and terminal outcomes.
- **Agents:** random, Rule-of-28 variants, one-turn Monte Carlo, Glenn–Aloi, Couto, Ridge, DDQN, and PPO.
- **RL formulation:** the current controlled setup learns Stop/Continue decisions while keeping Glenn–Aloi dice pairing fixed, isolating the strategic decision under study.
- **Features:** raw board state plus domain features such as exact next-roll bust probability, at-risk progress, completion threats, and opponent lead.
- **Evaluation:** alternating seats, deterministic independent RNG streams, fresh seeds, raw per-game records, and paired bootstrap intervals.

## Negative results are part of the evidence

The project also tests ideas that did **not** work:

```mermaid
xychart-beta
    title "Learned pairing diagnostic: PPO-451 versus pairing-only model"
    x-axis [Rule28-smart, MC]
    y-axis "Win rate (%)" 0 --> 70
    bar [64.6, 55.4]
    bar [50.4, 44.5]
```

The learned-pairing diagnostic lost to the fixed-pairing PPO by 14.1 and 10.9 percentage points, respectively, with paired intervals excluding zero. This prevents us from presenting “more learned components” as automatically better and motivates the current fixed-pairing design.

## Quick start

The reference implementation requires Python 3.10+ and has no third-party dependency for the simulator:

```bash
cd cant_stop_baselines
python3 -m unittest discover -s tests -t . -v
cant-stop --agent-a rule28-smart --agent-b random --games 100 --seed 42
```

Install the package from the repository root with:

```bash
python3 -m pip install -e .
```

Optional ML/RL dependencies and reproducible training commands are documented in [`cant_stop_baselines/README.md`](cant_stop_baselines/README.md).

## Repository guide

| Path | Purpose |
|---|---|
| [`src/cant_stop/`](src/cant_stop/) | Reusable rules, state model, agents, RL environment, and evaluation API |
| [`cant_stop_baselines/scripts/`](cant_stop_baselines/scripts/) | Training, dataset, evaluation, and analysis entry points |
| [`cant_stop_baselines/tests/`](cant_stop_baselines/tests/) | Rule, transition, reproducibility, agent, and training tests |
| [`cant_stop_baselines/docs/EXPERIMENT_STATUS.md`](cant_stop_baselines/docs/EXPERIMENT_STATUS.md) | Source of truth for results and research decisions |
| [`spec.pdf`](spec.pdf) | Course specification; not the engine contract |

## Reproducibility and limitations

Every formal run records configuration, seeds, checkpoints, evaluation summaries, and raw game-level results under local `runs/` and `results/` directories. Before interpreting a result, read the experiment status document: it distinguishes promoted evidence, screening runs, negative results, and incomplete experiments.

Known limitations include two-player games only, fixed dice pairing in the promoted Stop/Continue study, no graphical interface, and no claim of full-game MCTS. The simulator-generated games and held-out evaluation protocol are described in the experiment documentation.

## References and citations

- **[1]** Martin Keller, [“Can't Stop? Try The Rule of 28”](https://www.solitairelaboratory.com/abstract/dicedominoes.html). Source of the traditional Rule-of-28 baseline.
- **[2]** James Glenn and Christian Aloi, [*A Generalized Heuristic for Can't Stop*](https://cdn.aaai.org/ocs/123/123-2430-1-PB.pdf), FLAIRS 2009. Source of the GA pairing weights and published heuristic comparison.
- **[3]** Leandro C. Medeiros, David S. Aleixo, and Levi H. S. Levis, [*What Can We Learn Even From the Weakest? Learning Sketches for Programmatic Strategies*](https://doi.org/10.1609/aaai.v36i7.20744), AAAI 2022. Source of the Couto/Sketch-SA(O) comparison agent.
- **[4]** J. Cameron Carter, *Using Machine Learning to Optimize the Board Game Can't Stop* (2026), user-provided thesis. Used as an additional implementation and feature reference; the adapter is explicitly labeled in this repository.
- **[5]** This project, [experiment record](cant_stop_baselines/docs/EXPERIMENT_STATUS.md). Ridge, DDQN, PPO, and `mc-strong` are project implementations; their reported win rates come from this repository's seeded complete-game evaluations.

## License

MIT. See [`LICENSE`](LICENSE).
