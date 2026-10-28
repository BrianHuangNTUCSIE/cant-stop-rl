# Model guide: download or train locally

This project supports two ways to use the released learned agents:

| Goal | Recommended path |
|---|---|
| Run the project quickly or reproduce the published checkpoint | Download `best.pt` from Hugging Face |
| Change the architecture, seed, opponents, features, or hyperparameters | Train locally |
| Make a scientifically comparable experiment | Train locally and record a new run directory and seed |

The public model bundle contains three project-specific models: Double DQN,
PPO, and Ridge. The `.pt` files are PyTorch checkpoints and the `.joblib` file
is a scikit-learn-style artifact; none is a generic Transformers model.

## Option A: download the published model bundle

Use Python 3.10--3.13 and run these commands from the repository root:

```bash
cd cant_stop_baselines
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-rl.txt
# Also install this for the Ridge artifact:
python -m pip install -r requirements-ml.txt

huggingface-cli download \
  BrianHuang2003/cant-stop-ddqn \
  best.pt ppo/best.pt ridge/ridge_domain.joblib \
  config.json MODEL_FILES.json SHA256SUMS README.md \
  --repo-type model \
  --local-dir runs/hf_cant_stop_ddqn
```

Check the downloaded file before using it:

```bash
cd runs/hf_cant_stop_ddqn
shasum -a 256 -c SHA256SUMS
cd ../..
```

Evaluate it against the standard opponents:

```bash
python scripts/evaluate_ml.py \
  --models ddqn-hf=runs/hf_cant_stop_ddqn/best.pt \
  --opponents rule28-smart,mc \
  --games 1000 \
  --seed 700447 \
  --mc-rollouts 64 \
  --device cpu \
  --output-dir results/hf_ddqn_smoke
```

To evaluate the downloaded PPO checkpoint instead, change the model argument:

```bash
python scripts/evaluate_ml.py \
  --models ppo-hf=runs/hf_cant_stop_ddqn/ppo/best.pt \
  --opponents rule28-smart,mc --games 1000 --seed 700451 \
  --mc-rollouts 64 --device cpu \
  --output-dir results/hf_ppo_smoke
```

The Ridge artifact is evaluated through the same `--models` interface:

```bash
python scripts/evaluate_ml.py \
  --models ridge-hf=runs/hf_cant_stop_ddqn/ridge/ridge_domain.joblib \
  --opponents rule28-smart,mc --games 1000 --seed 700642 \
  --mc-rollouts 64 \
  --output-dir results/hf_ridge_smoke
```

Use an even number of games: the evaluator alternates seats and uses paired
dice seeds. For report-quality results, use held-out seeds, preserve the raw
records, and use the paired-bootstrap analysis rather than treating every game
as independent.

The checkpoint metadata and model card are available at:

<https://huggingface.co/BrianHuang2003/cant-stop-ddqn>

## Option B: train a model locally

Choose this path if you want to change the training seed, model architecture,
feature encoding, replay strategy, opponent mixture, or reward settings. All
generated checkpoints should remain under `runs/`; these files are intentionally
ignored by Git.

Install the optional RL dependencies first:

```bash
cd cant_stop_baselines
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-rl.txt
```

Run a short smoke experiment:

```bash
python scripts/train_dqn.py \
  --backend python \
  --algorithm double-dqn \
  --domain-features \
  --pairing-policy ga \
  --hidden-sizes 128,128 \
  --episodes 20000 \
  --device cpu \
  --seed 441 \
  --output-dir runs/ddqn_local_seed441
```

For the controlled reference configuration used by the project, see the full
command in [`cant_stop_baselines/README.md`](../cant_stop_baselines/README.md),
under “Ready-to-run controlled reference experiment”. It uses 100,000 episodes,
5-step returns, fixed Glenn--Aloi pairing, domain features, and validation
against `rule28-smart` and `mc`.

After training, evaluate the selected checkpoint:

```bash
python scripts/evaluate_ml.py \
  --models ddqn-local=runs/ddqn_local_seed441/best.pt \
  --opponents rule28-smart,mc \
  --games 1000 \
  --seed 700441 \
  --mc-rollouts 64 \
  --device cpu \
  --output-dir results/local_ddqn_seed441
```

Each training seed is a separate experiment. Do not compare a newly trained
model with the public checkpoint using only one convenient game batch; record
the code commit, configuration, training seed, evaluation seeds, dependencies,
and raw game-level results.

## Which option should I choose?

- **Just want to run the agent:** download from Hugging Face.
- **Want a fast demo:** download and run a small 100--1,000 game evaluation.
- **Want to reproduce or extend the research:** train locally with a new output
  directory and seed.
- **Want to cite the released result:** use the Hugging Face checkpoint and its
  `config.json` and checksum; do not silently replace it with a locally trained
  checkpoint.
