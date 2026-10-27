#!/usr/bin/env bash
set -euo pipefail

# Final confirmation run: same retained configurations, new independent seeds.
# Run model families sequentially to avoid oversubscribing the CPU.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run_ppo() {
  local seed="$1"
  .venv/bin/python3 scripts/train_ppo.py \
    --episodes 100000 --num-envs 8 --num-workers 8 \
    --backend python-multiprocess --rollout-steps 512 \
    --ppo-epochs 4 --minibatch-size 256 --learning-rate 0.0001 \
    --gamma 0.99 --gae-lambda 0.95 --clip-epsilon 0.2 \
    --entropy-coefficient 0.01 --value-coefficient 0.5 \
    --max-grad-norm 0.5 --hidden-sizes 128,128 --domain-features \
    --pairing-policy ga --opponent-mix mc:0.6,rule28-smart:0.4 \
    --training-mc-rollouts 16 --mc-rollouts 64 \
    --validation-opponents rule28-smart,mc --validation-games 500 \
    --eval-games 3000 --eval-every 5000 --seed "$seed" \
    --torch-threads 1 --output-dir "runs/ppo_mc_focused_seed${seed}"
}

run_ddqn() {
  local seed="$1"
  .venv/bin/python3 scripts/train_dqn.py \
    --algorithm double-dqn --domain-features --device cpu \
    --episodes 100000 --num-envs 16 --num-workers 16 \
    --backend python-multiprocess --hidden-sizes 128,128 \
    --network-type standard --pairing-policy ga --gamma 0.99 \
    --learning-rate 0.0003 --batch-size 256 --replay-capacity 500000 \
    --replay-type uniform --warmup-steps 10000 --train-frequency 4 \
    --target-update-steps 5000 --epsilon-start 1.0 --epsilon-end 0.03 \
    --epsilon-decay-steps 1500000 --gradient-clip 5.0 --n-step 5 \
    --opponent-mix rule28-smart:0.35,mc:0.35,rule28-ga-pure:0.1,rule28:0.1,random:0.1 \
    --training-mc-rollouts 16 --training-mc-future-roll-limit 20 \
    --counterfactual-loss-weight 0.1 --counterfactual-rollouts 4 \
    --counterfactual-every 16 --validation-opponents rule28-smart,mc \
    --validation-mc-rollouts 32 --validation-seed-blocks 1 \
    --checkpoint-metric minimum-win-rate --eval-games 5000 \
    --eval-every 5000 --checkpoint-every 5000 --seed "$seed" \
    --torch-threads 1 --output-dir "runs/ddqn_mc_counterfactual_seed${seed}"
}

run_ppo 460 & ppo_a=$!
run_ppo 461 & ppo_b=$!
wait "$ppo_a" "$ppo_b"

run_ddqn 448 & ddqn_a=$!
run_ddqn 449 & ddqn_b=$!
wait "$ddqn_a" "$ddqn_b"

.venv/bin/python3 scripts/train_ml_models.py \
  --dataset runs/multi_policy_counterfactual_dataset_seed20261028.csv \
  --models ridge --features domain --seed 642 \
  --output-dir runs/ridge_multi_policy_seed642
.venv/bin/python3 scripts/train_ml_models.py \
  --dataset runs/multi_policy_counterfactual_dataset_seed20261028.csv \
  --models ridge --features domain --seed 643 \
  --output-dir runs/ridge_multi_policy_seed643

echo "Final retraining complete."
