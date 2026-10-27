# Can't Stop — Python baseline simulator

A reproducible **two-player** implementation of *Can't Stop* for the CS 441 reinforcement-learning final project. The simulator uses Python **3.10+** with **no third-party dependencies**; DQN training additionally requires PyTorch.

## Run

From this folder:

```bash
python3 -m unittest discover -s tests -t . -v
cant-stop --agent-a rule28-smart --agent-b random --games 1000 --seed 42
cant-stop --agent-a couto --agent-b glenn-aloi --games 5000 --seed 42 --csv games.csv
cant-stop --agent-a rule28-smart --agent-b rule28 --games 1000 --seed 42 --csv games.csv
cant-stop --agent-a mc --agent-b rule28-smart --games 100 --mc-rollouts 32 --seed 42
cant-stop --agent-a vmerckle_monte --agent-b random --games 100 --mc-rollouts 16 --seed 42
cant-stop --agent-a vmerckle_monte_uct --agent-b random --games 100 --mc-rollouts 16 --seed 42
cant-stop --agent-a rule28-smart --agent-b random --games 100 --json
```

All output is deterministic for the same code, seed, and agent parameters. The optional CSV has one row per game.

`vmerckle_monte` is flat root Monte Carlo; `vmerckle_monte_uct` is a root-only
UCB1 bandit, not full-tree UCT/MCTS. Both use explicit `--mc-rollouts`
budgets and private random simulations.

Saved project-local Sketch JSON programs can be evaluated without enabling
Python code execution:

```bash
cant-stop --sketch-program-a programs/example.json \
  --agent-b rule28-smart --games 100 --seed 42
```

The JSON adapter is an explicitly labeled adaptation of the public Medeiros
DSL, not a claim that an article champion has been reproduced.

The public Carter Keras checkpoint can be supplied explicitly with
`--carter-checkpoint-a` or `--carter-checkpoint-b` when TensorFlow is installed.
Because its original simulator differs from this engine, this path is labeled
an adaptation and is not included in `AGENT_NAMES`.

For reproducible 100-game comparisons with raw records and configuration
metadata, use the evaluation-only matrix runner:

```bash
python3 scripts/evaluate_agent_comparison.py --output-dir results/agent-smoke
```

Add independent seed batches for between-seed variability with, for example,
`--replicates 5`; each replicate is written separately with its seed in the
metadata, including elapsed evaluation time.

Analyze those records with paired seed resampling rather than independent-game
intervals:

```bash
python3 scripts/analyze_paired_results.py results/agent-smoke
```

When the directory contains multiple `--replicates`, the same command also
reports descriptive between-seed mean and standard deviation summaries.

## Train DQN / Double DQN

### Simulator backends and profiling

The trainer keeps the original in-process implementation as the default
(`--backend python`, an alias for `python-reference`).  Environment stepping can
instead run in persistent worker processes while network inference, replay, and
optimization remain centralized:

```bash
python scripts/train_dqn.py --backend python-multiprocess \
  --num-envs 8 --num-workers 8 --episodes 20000 --seed 441
```

Workers own independent seeded environments and receive batched actions. A
worker crash is reported to the learner, and each policy/environment RNG remains
separate. Run the reproducible Phase-0 benchmark before selecting a worker
count:

```bash
python scripts/benchmark_backends.py --backend all --workers 4 \
  --episodes 100 --iterations 10000 \
  --output runs/backend_benchmark_seed441.json
```

The JSON reports primitive operation rates, environment episodes/transitions
per second, and deterministic checksums. Compare worker counts `1,2,4,8`; more
workers can be slower when MC opponents or Torch thread pools oversubscribe the
CPU.

### Ready-to-run controlled reference experiment

The validated starting point is the in-process Python backend. Keep the GA
pairing fixed when the experiment is intended to isolate Stop/Continue learning;
use a unique output directory for every seed:

```bash
python scripts/train_dqn.py \
  --backend python --algorithm double-dqn --pairing-policy ga \
  --domain-features --n-step 5 --hidden-sizes 128,128 \
  --episodes 100000 --num-envs 16 --torch-threads 1 \
  --epsilon-decay-steps 1500000 --epsilon-end 0.03 \
  --replay-capacity 500000 --training-mc-rollouts 16 \
  --validation-opponents rule28-smart,mc \
  --validation-mc-rollouts 32 --eval-every 5000 \
  --interim-eval-games 1000 --eval-games 5000 \
  --checkpoint-every 5000 --device cpu --seed 441 \
  --output-dir runs/ddqn_reference_seed441
```

The training, validation, checkpoint, and metrics path was verified with a
short smoke run. Do not use `python-multiprocess` for the formal run: current
benchmarks make it slower than the reference backend while producing the same
correctness checksum.

Use a Python version supported by PyTorch (rather than a just-released Python
version for which wheels may not exist), then install the RL dependency:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-rl.txt
python scripts/train_dqn.py --algorithm double-dqn --enhanced-features \
  --hidden-sizes 128,128 --episodes 20000 --device cpu --seed 441 \
  --output-dir runs/ddqn_enhanced_128_seed441
```

Robust training against the baseline distribution uses mixed opponents,
domain-informed features, 5-step returns, and minimax checkpoint selection:

```bash
python scripts/train_dqn.py \
  --algorithm double-dqn --domain-features --n-step 5 \
  --episodes 100000 --num-envs 16 --hidden-sizes 128,128 \
  --epsilon-decay-steps 1500000 --epsilon-end 0.03 \
  --replay-capacity 500000 \
  --opponent-mix rule28-smart:0.35,mc:0.35,rule28-ga-pure:0.1,rule28:0.1,random:0.1 \
  --training-mc-rollouts 16 \
  --validation-opponents rule28-smart,mc --validation-mc-rollouts 32 \
  --eval-every 5000 --interim-eval-games 1000 --eval-games 5000 \
  --checkpoint-every 5000 --device cpu --seed 441 \
  --output-dir runs/ddqn_robust_domain_n5_seed441
```

The training opponent for each episode is sampled deterministically from
`--opponent-mix`. `best.pt` maximizes the **minimum** validation win rate over
`--validation-opponents`, rather than specializing checkpoint selection to one
baseline. Training and validation Monte Carlo rollout counts are separate so
training can use a cheaper approximation.

The next controlled replay ablation uses proportional Prioritized Experience
Replay (PER). It keeps the standard Double DQN/domain-feature configuration
unchanged and only replaces uniform replay. PER uses `alpha=0.6`, anneals the
importance-sampling `beta` value from `0.4` to `1.0`, and records these values in
`config.json`:

```bash
python scripts/train_dqn.py \
  --algorithm double-dqn --domain-features --n-step 5 \
  --network-type standard --hidden-sizes 128,128 \
  --replay-type prioritized --per-alpha 0.6 \
  --per-beta-start 0.4 --per-beta-end 1.0 \
  --episodes 100000 --num-envs 16 --torch-threads 8 \
  --epsilon-decay-steps 1500000 --epsilon-end 0.03 \
  --replay-capacity 500000 --training-mc-rollouts 16 \
  --validation-opponents rule28-smart,mc --validation-mc-rollouts 32 \
  --eval-every 5000 --interim-eval-games 1000 --eval-games 5000 \
  --checkpoint-every 5000 --device cpu --seed 441 \
  --output-dir runs/ddqn_per_seed441_threads8
```

`--replay-type uniform` remains the default, so older commands and control
experiments are unchanged. On CPU servers, benchmark `--torch-threads 4` and
`8`; the small network and Python environment loop do not necessarily scale
linearly with more threads.

### Frozen-snapshot self-play

Self-play training is available with `--self-play`. The learner plays against
frozen snapshots of itself; snapshots are added periodically and sampled from
a bounded history, which is more stable than playing against a network that is
updated during the same game. The validation opponents remain external
policies, so self-play win rate is not used as the performance claim:

```bash
python scripts/train_dqn.py \
  --self-play --self-play-snapshot-every 5000 \
  --self-play-history-size 20 --self-play-latest-probability 0.5 \
  --algorithm double-dqn --domain-features --pairing-policy ga \
  --hidden-sizes 128,128 --n-step 5 --episodes 100000 --num-envs 16 \
  --epsilon-decay-steps 1500000 --epsilon-end 0.03 \
  --replay-capacity 500000 --counterfactual-loss-weight 0.05 \
  --counterfactual-rollouts 4 --counterfactual-every 16 \
  --validation-opponents rule28-smart,mc --validation-mc-rollouts 32 \
  --eval-every 5000 --interim-eval-games 1000 --eval-games 5000 \
  --checkpoint-every 5000 --device cpu --seed 447 \
  --output-dir runs/ddqn_self_play_seed447
```

`--self-play-latest-probability` controls how often the newest snapshot is
used; otherwise an older snapshot is sampled. `--self-play-history-size`
limits the retained in-memory opponent pool. Compare the resulting
`best.pt` against Ridge, `rule28-smart`, `mc`, and the retained DDQN control
using paired held-out games.

Reward shaping is also opt-in. The original banked-progress potential remains
the default; `--shaping-profile race-aware` adds completion-threat and
temporary-progress terms while preserving potential-based shaping:

```bash
python scripts/train_dqn.py \
  --algorithm double-dqn --domain-features --n-step 5 \
  --reward-shaping 0.25 --shaping-profile race-aware \
  --episodes 100000 --seed 441 \
  --output-dir runs/ddqn_race_aware_shaping_seed441
```

This only prepares a run; it does not change existing checkpoints or launch an
experiment automatically.

### Joint learned pairing experiment

The fixed-pairing controls remain unchanged. To train a separate agent that
selects one of the legal dice pairings and then chooses Stop/Continue, use
`--pairing-policy learned`. This uses 8 actions (six pairing slots plus
Continue/Stop) and masks illegal actions:

```bash
python scripts/train_dqn.py \
  --algorithm double-dqn --pairing-policy learned \
  --domain-features --n-step 5 --episodes 100000 --num-envs 16 \
  --training-mc-rollouts 64 --validation-mc-rollouts 64 \
  --validation-opponents rule28-smart,mc \
  --eval-every 5000 --interim-eval-games 1000 --eval-games 5000 \
  --device cpu --seed 441 \
  --output-dir runs/ddqn_joint_pairing_seed441
```

Joint-mode observations are 86-dimensional with domain features (68 state
features plus 18 pairing-candidate features). Evaluate the resulting
checkpoint with the existing `evaluate_dqn.py` command; the checkpoint records
`pairing_policy: "learned"` and uses the learned pairing during evaluation.

For faster early training, a pretrained fixed-pairing DQN can be used as the
exclusive opponent for an initial curriculum phase. After the warm-up, the
normal `--opponent-mix` is used automatically:

```bash
python scripts/train_dqn.py \
  --algorithm double-dqn --pairing-policy learned \
  --domain-features --n-step 5 --episodes 100000 --num-envs 16 \
  --dqn-opponent runs/ddqn_mc_counterfactual_seed447/best.pt \
  --dqn-warmup-episodes 10000 \
  --training-mc-rollouts 64 --validation-mc-rollouts 64 \
  --validation-opponents rule28-smart,mc \
  --eval-every 5000 --interim-eval-games 1000 --eval-games 5000 \
  --device cpu --seed 441 \
  --output-dir runs/ddqn_joint_pairing_warmup10k_seed441
```

The attempted seed-441 joint-pairing runs were stopped before a complete
predeclared evaluation and are not evidence of improvement. For an all-fast
diagnostic, set `--dqn-warmup-episodes` equal to `--episodes` and use
`--validation-opponents dqn-opponent`; this removes MC from both training and
validation, but it no longer measures MC generalization.

The revised DDQN v2 experiment is a controlled **depth-only ablation**. It uses
the successful v1 configuration unchanged and compares `128,128`, `256,256`,
and `256,256,128` with the same three training seeds (`441,442,443`). It does
not use strategic features, dueling heads, reward shaping, Couto pairing, or a
strong curriculum. Start it in the background with:

```bash
./scripts/run_ddqn_v2_training.sh --smoke  # verify the pipeline first
./scripts/run_ddqn_v2_training.sh          # 3 architectures x 3 seeds x 100k episodes
tail -f runs/ddqn_v2_depth_ablation.log
./scripts/run_ddqn_v2_training.sh --stop   # stop the launcher and current child run
```

Results are written under `runs/ddqn_v2_depth_ablation/`, including one
`multi_seed_summary.json` per architecture and combined
`depth_ablation_summary.{json,csv}` files. Validation logs win rate,
busts/turn, stops/turn, rolls/turn, and greedy stop rate. After a 15,000-episode
grace period, a run stops early when aggregate validation busts/turn exceeds
`0.55` twice consecutively; its best checkpoint and diagnostic checkpoint are
retained.

Run the same configuration over several independent training seeds:

```bash
python scripts/train_multi_seed.py \
  --seeds 441,442,443,444,445 \
  --output-root runs/ddqn_robust_5seeds \
  --algorithm double-dqn --domain-features --n-step 5 \
  --episodes 100000 --epsilon-decay-steps 1500000 --epsilon-end 0.03 \
  --replay-capacity 500000 --eval-every 5000 \
  --interim-eval-games 1000 --eval-games 5000 --device cpu
```

`scripts/train_multi_seed.py` runs seeds sequentially, forwards its unknown arguments
to `scripts/train_dqn.py`, and writes `multi_seed_summary.json` with per-seed and
between-seed results.

`--device cpu` is the training default because the small 128x128 network is
typically faster there than on Apple MPS once dispatch and tensor-transfer
overhead are included. Use `--device auto` to prefer an available accelerator,
or explicitly select `cuda`, `mps`, or `xpu`; unavailable explicit devices fail
clearly. Checkpoints store CPU weights and can be loaded on a different
supported device:

```python
from cant_stop.dqn import DQNAgent

agent = DQNAgent.from_checkpoint(
    "runs/ddqn_enhanced_128_seed441/best.pt", device="auto"
)
```

Evaluate a checkpoint against every baseline and print a compact result table:

```bash
python scripts/evaluate_dqn.py \
  --checkpoint runs/ddqn_enhanced_128_cpu_fast_seed441/final.pt \
  --games 5000 --seed 30000441 --device cpu
```

With no `--opponent`, this evaluates all eight built-in agents (`random`,
`rule28`, `rule28-smart`, `rule28-ga-pure`, `mc`, `glenn-aloi`, `couto`, and
`couto-ga-stop`). It does not save files unless
`--summary PATH` or `--csv PATH` is explicitly supplied with one
`--opponent NAME`. The random baseline chooses pairings randomly and stops with
probability 0.3; override the latter with `--random-stop-probability`.

Training defaults to 16 interleaved environments, a 250,000-transition tensor
ring replay buffer, batch size 256, and one update per four collected
transitions. The conservative stability defaults use a `3e-4` learning rate,
a 10,000-step warmup, a 5,000-step hard target update, and epsilon decay over
300,000 transitions. The default opponent mixture is 35% `rule28-smart`, 35%
low-cost `mc`, and 10% each of `rule28-ga-pure`, `rule28`, and `random`.
The default return remains one-step for backwards-compatible ablations; use
`--n-step 5` for the robust configuration above.
Override these values explicitly for ablations.
Each run writes `config.json`, incremental episode and throughput CSVs, periodic
and best checkpoints, greedy validation results, and separate held-out reports
for the final and best-validation checkpoints. `evaluation_summary.json` and
`evaluation_games.csv` describe `final.pt`; `best_evaluation_summary.json` and
`best_evaluation_games.csv` describe `best.pt`. Training alternates the
learner's seat and validates against `rule28-smart` and `mc` with fixed GA
pairing. Summary JSON files contain one report per validation opponent and the
minimum-win-rate selection score.

### Phase-A counterfactual PPO

This controlled experiment keeps the successful GA dice pairing fixed and
transfers PPO-451's 68-feature Stop/Continue policy into the 80-feature
strategic representation. Every eighth eligible decision receives an MC
counterfactual label `(Q_continue, Q_stop)`; PPO retains terminal rewards and
adds a weighted squared loss on the action margin. This is auxiliary
supervision, not pure MC imitation. The smoke test passed with the same code;
the following is the predeclared training run:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python3 scripts/train_ppo.py \
  --pairing-policy ga \
  --strategic-features \
  --initial-ppo runs/ppo_mc_focused_seed451/top_1.pt \
  --counterfactual-loss-weight 0.10 \
  --counterfactual-rollouts 8 \
  --counterfactual-every 8 \
  --counterfactual-future-roll-limit 16 \
  --episodes 100000 --num-envs 8 --backend python \
  --rollout-steps 512 --ppo-epochs 4 --minibatch-size 256 \
  --learning-rate 1e-4 \
  --opponent-mix mc:0.6,rule28-smart:0.4 \
  --training-mc-rollouts 16 --mc-rollouts 64 --mc-future-roll-limit 20 \
  --validation-opponents rule28-smart,mc \
  --validation-games 500 --eval-games 3000 --eval-every 5000 \
  --progress-every 1000 --seed 454 \
  --output-dir runs/ppo_counterfactual_seed454
```

Use this run only as Phase A. Stop it if the minimum validation win rate is
below 52% at 30,000 episodes. A surviving checkpoint must be evaluated against
MC, Ridge, and Rule28-smart on fresh common held-out seeds before any claim of
improvement. Do not add learned pairing or Ridge curriculum to this run.
The default output directory includes the algorithm, feature set, and seed.
PyTorch requests deterministic kernels and records the selected hardware, but
bit-for-bit equality across different accelerator types is not guaranteed.

## Classical supervised ML experiments

The traditional ML pipeline independently labels decision states instead of
copying the DDQN policy. For each sampled state it forces Stop and Continue,
runs both branches with the same future random seeds, and estimates each
action's full-game win probability. After the forced action, both seats use
`rule28-smart`; these are therefore approximate rollout targets, not claims of
optimal play.

Run the complete report-scale pipeline in the background with one command:

```bash
./scripts/run_classical_ml_experiments.sh
```

It generates 10,000 states with 32 paired rollouts per action, trains all five
model families on all three feature encodings, then evaluates the five domain
models for 5,000 games against both `rule28-smart` and `mc`. Follow progress and
find results with:

```bash
tail -f runs/classical_ml_seed541.log
cat runs/classical_ml_seed541/models/training_report.json
cat runs/classical_ml_seed541/evaluation/evaluation_summary.json
```

The launcher uses one seed per stage, runs under `caffeinate`, records its PID
in `runs/classical_ml_seed541.pid`, and refuses to overwrite an existing run.
The individual commands below remain available for smaller tests and ablations.

Install the optional dependency and generate one dataset containing all 68
features (the first 58 and 60 values are the raw and enhanced encodings):

```bash
python -m pip install -r requirements-ml.txt
python scripts/generate_ml_dataset.py \
  --games 1000 --max-states 10000 --rollouts 16 --seed 541 \
  --output data/generated/classical_ml_dataset.csv
```

Rollout labelling is the expensive stage. First use a small smoke test such as
`--games 20 --max-states 100 --rollouts 2`; use more states and at least 16–64
paired rollouts for report-quality data. Dataset metadata is saved next to the
CSV. Collection uses exploratory Stop/Continue actions and fixed GA pairing.

Train Logistic Regression, Decision Tree, Random Forest, Linear SVM, and Ridge
regression on raw, enhanced, and domain features:

```bash
python scripts/train_ml_models.py \
  --dataset data/generated/classical_ml_dataset.csv \
  --models all --features all --seed 641 \
  --output-dir runs/classical_ml/models
```

Ridge predicts `Q(continue) - Q(stop)` and continues only when the prediction is
positive. The classifiers predict action 0 (Continue) or 1 (Stop). Source games,
not individual states, are divided into 70/15/15 train/validation/test splits.
Hyperparameters minimize validation rollout regret, and `training_report.json`
contains untouched test metrics and confusion matrices.

Evaluate selected artifacts in complete games against identical held-out seeds:

```bash
python scripts/evaluate_ml.py \
  --models \
    runs/classical_ml/models/logistic_domain.joblib \
    runs/classical_ml/models/decision-tree_domain.joblib \
    runs/classical_ml/models/random-forest_domain.joblib \
    runs/classical_ml/models/linear-svm_domain.joblib \
    runs/classical_ml/models/ridge_domain.joblib \
  --opponents rule28-smart,mc --games 5000 --seed 40000641 \
  --mc-rollouts 32 --workers 4 \
  --output-dir results/classical_ml/evaluation
```

Evaluation alternates seats and reports a confidence interval bootstrapped over
the shared-dice two-game seed pairs. For models evaluated in the same command,
the summary also contains paired bootstrap intervals for pairwise win-rate
differences against each opponent. It saves raw per-game CSV files. Keep this
evaluation seed disjoint from dataset collection and model selection.
`--workers N` evaluates up to N model-opponent matchups in separate processes;
results and raw records retain deterministic command-order output. Leave it at
the default of 1 for cheap matchups where process startup would dominate.

To diagnose the fixed pairing policy without retraining PPO, load the same
checkpoint under distinct aliases and apply model-specific `ga` and `couto`
pairing overrides. The PPO Stop/Continue network remains frozen:

```bash
python scripts/evaluate_ml.py \
  --models ppo-ga=runs/ppo/best.pt ppo-couto=runs/ppo/best.pt \
  --ppo-pairing ppo-ga=ga \
  --ppo-pairing ppo-couto=couto \
  --opponents rule28-smart,mc --games 3000 --seed 900451 \
  --mc-rollouts 64 --bootstrap-samples 5000 --workers 4 \
  --output-dir runs/ppo_pairing_diagnostic
```

Train the transfer-initialized joint PPO with six masked pairing actions and
Continue/Stop actions 6--7. The 68-input PPO trunk, value head, and
Continue/Stop outputs are copied from PPO-451; 20,000 deterministic GA-labeled
pair states initialize the 18 new pairing-input columns and six new pairing
outputs before joint PPO updates, while preserving PPO-451 on decision states:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python3 scripts/train_ppo.py \
  --pairing-policy learned \
  --initial-ppo runs/ppo_mc_focused_seed451/top_1.pt \
  --pairing-imitation-examples 20000 --pairing-imitation-epochs 3 \
  --episodes 100000 --num-envs 8 --backend python \
  --rollout-steps 512 --ppo-epochs 4 --minibatch-size 256 \
  --learning-rate 1e-4 \
  --opponent-mix mc:0.6,rule28-smart:0.4 \
  --training-mc-rollouts 16 --mc-rollouts 64 \
  --mc-future-roll-limit 20 \
  --validation-opponents rule28-smart,mc \
  --validation-games 500 --eval-games 3000 --eval-every 5000 \
  --progress-every 1000 --seed 452 \
  --output-dir runs/ppo_joint_pairing_seed452_v2
```

This is one training seed. Its automatically selected checkpoints are screening
results, not final evidence; compare the leading checkpoint with PPO-451 and
the retained DDQN on a fresh common evaluation seed before promotion.

### Pairing-only diagnostic experiment

The next diagnostic freezes PPO-451's Stop/Continue policy and trains only the
new pairing input columns and six pairing output rows. PPO gradients for the
original 68 state inputs and Continue/Stop outputs are explicitly masked. This
isolates whether learned dice pairing helps without allowing it to destroy the
strong fixed-pairing policy:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python3 scripts/train_ppo.py \
  --pairing-policy learned \
  --initial-ppo runs/ppo_mc_focused_seed451/top_1.pt \
  --freeze-decision-policy \
  --pairing-imitation-examples 20000 --pairing-imitation-epochs 3 \
  --episodes 100000 --num-envs 8 --backend python \
  --rollout-steps 512 --ppo-epochs 4 --minibatch-size 256 \
  --learning-rate 1e-4 \
  --opponent-mix mc:0.6,rule28-smart:0.4 \
  --training-mc-rollouts 16 --mc-rollouts 64 \
  --mc-future-roll-limit 20 \
  --validation-opponents rule28-smart,mc \
  --validation-games 500 --eval-games 3000 --eval-every 5000 \
  --progress-every 1000 --seed 453 \
  --output-dir runs/ppo_pairing_only_seed453
```

This is a diagnostic training seed, not evidence of superiority. If its
Stop/Continue behavior remains equivalent to PPO-451 but win rates do not
improve, retain PPO-451 with GA pairing and stop the learned-pairing branch.

Run DQN checkpoints and classical ML models against each other in one all-pairs
tournament. Use `NAME=PATH` labels when several checkpoints share a filename
such as `best.pt`:

```bash
python scripts/evaluate_ml_head_to_head.py \
  --dqn-models \
    ddqn-reference=runs/ddqn_mc_counterfactual_seed447/best.pt \
  --models \
    logistic=runs/classical_ml_seed541/models/logistic_domain.joblib \
    random-forest=runs/classical_ml_seed541/models/random-forest_domain.joblib \
  --games 5000 --seed 50000641 --device cpu \
  --workers 4 \
  --output-dir results/model_head_to_head
```

Every unordered pair plays `--games` games with alternating seats; each
two-game seat-swapped pair shares a dice seed. The command writes one raw CSV
per matchup, `win_rate_matrix.csv`, and `head_to_head_summary.json`. At least
two total models are required. Plain paths are also accepted when their file
stems are unique. During long runs it reports games completed, throughput, and
an ETA every 50 games; change this with `--progress-every N` or pass
`--progress-every 0` to disable progress output. `--workers N` evaluates up to
N different matchups in separate processes and reports whenever a matchup
finishes. Parallelism helps when there are at least three models; one matchup
cannot be split across workers. Start around half of the machine's performance
cores to avoid memory pressure and nested numerical-library contention.

Built-in agents can also be included without model files. For example, this
runs a reproducible tournament over the two published strong heuristics, the
fixed-pairing Couto adaptation, and the project's smart Rule-of-28 baseline:

```bash
python scripts/evaluate_ml_head_to_head.py \
  --agents glenn-aloi couto couto-ga-stop rule28-smart \
  --games 5000 --seed 50000641 --workers 4 \
  --output-dir results/strong_agent_tournament
```

For a one-command six-agent tournament containing the strongest saved DDQN,
the strongest saved classical model, Couto, its fixed-pairing adaptation,
Glenn–Aloi, and MC, run:

```bash
./scripts/run_champions_tournament.sh
```

The launcher creates a Python 3.11 `.venv` and installs missing dependencies
when necessary, then runs 15 matchups × 5,000 games in the background under
`caffeinate`. Use `./scripts/run_champions_tournament.sh --smoke` for 100 games per
matchup. Progress is written to `runs/champions_tournament.log` and the PID to
`runs/champions_tournament.pid`; it refuses to overwrite an existing result
directory.

## File structure

All python entry-point scripts and shell launchers live under `scripts/` and are
invoked from this folder (for example `python scripts/train_dqn.py ...` or
`./scripts/run_champions_tournament.sh`); each entry script extends its import
path to the project root so relative `runs/...` output paths keep their previous
meaning.

- `cant-stop`: backwards-compatible import facade and evaluation CLI entrypoint.
- `cant_stop/constants.py`: board dimensions and heuristic constants.
- `cant_stop/model.py`: immutable `Move` and agent-facing `State` values/features.
- `cant_stop/rules.py`: pure move generation, bust probability, and heuristic scoring.
- `cant_stop/game.py`: policy-free mutable game state machine.
- `cant_stop/agents.py`: random, Rule-of-28, Glenn–Aloi, Couto Sketch-SA(O), and one-turn Monte Carlo agents.
- `cant_stop/evaluation.py`: seeded game execution and head-to-head metrics.
- `cant_stop/rl_env.py`: dependency-free Stop/Continue RL environment.
- `cant_stop/backend.py`, `config.py`, `transition.py`: versioned backend
  boundary, runtime metadata, and fixed actor-to-learner transition schema.
- `cant_stop/dqn.py`: PyTorch DQN/Double DQN, replay, device selection, and agent adapter.
- `cant_stop/classical_ml.py`: persisted sklearn policy adapter.
- `cant_stop/cli.py`: argument parsing and CSV/JSON output.
- `scripts/train_dqn.py`: reproducible training, checkpointing, and held-out evaluation CLI.
- `scripts/benchmark_backends.py`: deterministic simulator/actor throughput benchmark.
- `rollout/`: persistent multiprocessing actors and pickle-safe batch protocol.
- `learners/dqn_learner.py`: environment-independent replay/optimization owner.
- `scripts/train_multi_seed.py`: sequential/parallel multi-seed launcher and aggregation.
- `scripts/summarize_depth_ablation.py`: depth-ablation summary aggregation.
- `scripts/generate_ml_dataset.py`: paired full-game rollout data collection and labels.
- `scripts/train_ml_models.py`: grouped model selection and offline test metrics.
- `scripts/evaluate_dqn.py`: evaluate a saved DQN checkpoint against baselines.
- `scripts/evaluate_ml.py`: complete-game evaluation with pair-level bootstrap intervals.
- `scripts/evaluate_ml_head_to_head.py`: all-pairs DQN/classical-ML tournament and win-rate matrix.
- `scripts/run_*.sh`: one-command background launchers for the main experiment pipelines (`run_experiments.sh`, `run_classical_ml_experiments.sh`, `run_ddqn_v2_training.sh`, `run_ddqn_dueling_seed441.sh`, `run_ddqn_mc_curriculum.sh`, `run_champions_tournament.sh`).
- `tests/`: regression suite.
  - `test_cant_stop.py`: original rule and reproducibility regression tests.
  - `test_rules_extended.py`: exhaustive dice-outcome checks against an independent oracle and game invariants.
  - `test_evaluation_extended.py`: turn-limit, paired-seed, seat-alternation, and raw-record aggregation tests.
  - `test_dqn.py`, `test_rl_env.py`, `test_classical_ml.py`, `test_head_to_head.py`: ML component tests.
- `docs/EXPERIMENT_STATUS.md`: source of truth for completed experiment results and next steps.
- `docs/THIRD_PARTY_NOTICES.md`: license notices for the ported Sketch-SA(O) reference.
- `data/example_results.csv`: sample results of 100 games (`rule28-smart` vs `rule28`; seed 123). These are demonstration results, not formal research claims.
- `runs/`: model checkpoints and model-training artifacts (gitignored).
- `results/`: evaluation summaries and raw per-game records (gitignored).

## Gameplay rules implemented

- The standard 11 columns 2–12 with heights 3,5,7,9,11,13,11,9,7,5,3.
- Four fair independent six-sided dice; at most three distinct pair partitions.
- Must use both pair sums if legal. If both individually legal but only one free neutral marker remains, player may choose either one. If only one sum is legal, use that one. If neither is possible for all partitions, bust.
- When two pair sums coincide, a runner can advance twice (but only up to the top). Markers are limited to **three different active columns** per turn.
- Each turn starts with a mandatory roll. After a successful roll and move, the player chooses `stop` (bank) or `continue` (roll again).
- Bust loses **current-turn temporary progress only**. Previously banked progress is retained.
- Columns reached at their top become *owned and closed only after stopping*. A third owned column wins immediately.
- Implementation is **2 players only**, but the game mechanics extend naturally to 3–4 players.

## Agent variants

| CLI name | Pair selection | Stop / Continue policy |
|---|---|---|
| `random` | Random legal move | Stop with probability `0.3` after each successful roll |
| `rule28` | Classical center preference (`6-abs(7-col)`, new runner penalty 6) | Pure Keller Rule of 28: at least 3 active runners, score >=28 |
| `rule28-ga-pure` | Glenn & Aloi (2009) GA constant *move* weights | Same pure Rule of 28 |
| `rule28-smart` | GA constant move weights | Stop at >=28 with 3 markers **or** as soon as any column temporarily reaches the top |
| `mc` | GA constant move weights | One-turn Monte Carlo rollout estimate vs banking now; rollout policy = `rule28-smart` |
| `glenn-aloi` | Full released Glenn–Aloi optimized pairing | Full optimized progress/difficulty score and threshold 29 |
| `couto` | Original Sketch-SA(O), including secured board progress | Original synthesized quadratic-difficulty Stop/Continue policy |
| `couto-ga-stop` | Fixed GA constant move weights | Couto Stop/Continue policy only (**project adaptation**) |

**Important:** `rule28-smart` remains an explicit, simple hybrid baseline and is
not Glenn & Aloi's full agent. Use `glenn-aloi` for the released full heuristic.
`couto` ports the complete Sketch-SA(O) strategy; `couto-ga-stop` deliberately
replaces its pairing decision so Stop/Continue can be compared under the same
fixed pairing policy. Neither strong heuristic requires training.

GA constant dice-pair move weights (Glenn & Aloi, Table 1):

```
column:  2  3  4  5  6  7  8  9 10 11 12
weight:  7  0  2  0  4  3  4  0  2  0  7
```

Each new neutral marker used in the move costs 6 points. Both move weights and penalty are part of the GA constant-weight champion. **The weight for column 7 is 3**, as in the original paper.

Original Rule-of-28 scoring: opening a marker adds twice the per-advance column weight; later advances add the per-advance weight; with three active columns all odd +2, all even -2, all <8 +4, all >6 +4. The pure variant does not bank until all three neutral markers are used (unless game state is impossible due to losing, in which case engine handles bust).

## Use from Python

```python
from cant_stop import evaluate, Rule28Agent, RandomAgent, MonteCarloAgent

report = evaluate(
    Rule28Agent(pair_heuristic="ga", smart=True),
    RandomAgent(stop_probability=0.30),
    games=1000,
    seed=441,
    include_game_records=True,
)
print(report["agents"])
print(report["game_records"][0])
```

Implementing your own agent:

```python
from cant_stop import Move, State, pair_score

class MyAgent:
    def choose_move(self, state: State, moves: tuple[Move, ...], rng):
        return max(moves, key=lambda m: pair_score(state, m, "ga"))

    def should_stop(self, state: State, rng):
        # Replace with your DQN action selection.
        # 0 = continue; 1 = stop (or pick your own mapping consistently).
        features = state.features(enhanced=True)
        assert len(features) == 60
        return len(state.active) == 3 and state.rolls_this_turn >= 3
```

**How to integrate your RL agent:** Only implement `should_stop(...)` for stop/continue, while inheriting or copying `choose_move(...)` from `Rule28Agent`. `State` is immutable. `state.features()` produces **58 raw features**, and `state.features(enhanced=True)` produces **60**, adding exact one-roll bust probability (enumerating 1296 outcomes) and normalized at-risk progress. `state.features(domain=True)` produces **68** values by additionally exposing active/top marker counts, post-stop completions and immediate-win status, board-progress lead, risk-adjusted turn progress, normalized Rule-of-28 score, and opponent two-column threat. `state.features(strategic=True)` produces **80**, adding per-column unbanked progress and an explicit near-completion threat count. Earlier discussion had incorrectly summed 55 + 3 as 57; the correct raw dimension count is 58.

**State machine API for custom training loops:** `Game.snapshot()`, `Game.roll(rng)`, `Game.apply_move(move)`, `Game.stop()`, `Game.continue_turn()`; `game.phase` is one of `roll`, `pair`, `decision`, `done`. The evaluator is separate from the state machine. The state representation is relative to the *current player*, so custom loops must take care of terminal outcomes and opponent turns when calculating transitions/rewards.

The higher-level `StopContinueEnv` automates dice pairing and the fixed opponent,
and advances from one learner decision to the next. By default it uses terminal-only
rewards (`+1` win, `-1` loss, `0` otherwise). The optional `reward_shaping` scale
adds a potential-based banked-progress difference while preserving the terminal
objective. It uses Gymnasium-style return values without a Gymnasium dependency:

```python
from cant_stop import StopContinueAction, StopContinueEnv

env = StopContinueEnv(enhanced_features=True, learner_player=0, seed=441)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(
    StopContinueAction.CONTINUE
)
```

## Evaluation

`evaluate` alternates which physical seat each submitted agent occupies, so A and B each start approximately half the games. Consecutive pairs use the same dice seed (`seed + i//2`). Agent randomness uses deterministic per-seat streams separate from dice RNG. Game outcomes under seat swaps are **paired** and thus not independent samples; the displayed 95% Wilson intervals should be viewed as descriptive approximations, **not** definitive paired confidence intervals. For serious reports, evaluate multiple independent base seeds and bootstrap on paired seeds or report between-seed variability.

Enhanced-state bust probability remains exact over all 1,296 ordered rolls, but
is implemented with a cached 11-bit mask of currently stepable columns. This is
equivalent to enumerating `legal_moves` for every roll without repeating that
Python-heavy work for every full board position.

Report includes win rates, approximate Wilson 95% intervals, seat-specific wins, rolls/game, turns/game, busts/turn, and stops. `--csv PATH` exports raw game-level data for analysis. If max turns is exceeded, the game is recorded as a draw.

**Monte Carlo agent limitations:** Rollouts simulate just the remainder of the current turn, not complete two-player games; future moves in each rollout use GA pairing plus smart Rule of 28 stopping. The utility is normalized banked progress plus bonuses for claimed/winning columns. It is deliberately an **approximate one-turn Monte Carlo baseline**, not MCTS and not an optimal agent. `--mc-rollouts` raises accuracy and runtime together.

## Sources

- Martin Keller, “Can't Stop? Try The Rule of 28” (1986), https://www.solitairelaboratory.com/abstract/dicedominoes.html
- James Glenn and Christian Aloi, *A Generalized Heuristic for Can't Stop*, FLAIRS 2009, https://cdn.aaai.org/ocs/123/123-2430-1-PB.pdf (Table 1 for GA constant move weights).
- Leandro C. Medeiros, David S. Aleixo, and Levi H. S. Levis, *What Can We Learn Even From the Weakest? Learning Sketches for Programmatic Strategies*, AAAI 2022, https://doi.org/10.1609/aaai.v36i7.20744; MIT-licensed reference implementation: https://github.com/leandrocouto/sketch-learning. See `docs/THIRD_PARTY_NOTICES.md`.
- J. Cameron Carter, *Using Machine Learning to Optimize the Board Game Can't Stop* (2026), user-provided thesis, for Rule-of-28 scoring and baseline ideas.

## Not implemented

- Graphical user interface, >2 players, rule variants, or a full MCTS agent.
- Full-game MCTS, learned joint dice-pair selection, or a graphical interface.
