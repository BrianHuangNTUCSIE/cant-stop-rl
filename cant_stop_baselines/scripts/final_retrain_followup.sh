#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="runs/final_retrain_followup_20261027.log"

for path in \
  runs/ppo_mc_focused_seed460/top_1.pt \
  runs/ppo_mc_focused_seed461/top_1.pt \
  runs/ddqn_mc_counterfactual_seed448/best.pt \
  runs/ddqn_mc_counterfactual_seed449/best.pt \
  runs/ridge_multi_policy_seed642/ridge_domain.joblib \
  runs/ridge_multi_policy_seed643/ridge_domain.joblib; do
  while [[ ! -f "$path" ]]; do
    sleep 60
  done
done

.venv/bin/python3 scripts/evaluate_ml.py \
  --models \
    ppo460=runs/ppo_mc_focused_seed460/top_1.pt \
    ppo461=runs/ppo_mc_focused_seed461/top_1.pt \
    ddqn448=runs/ddqn_mc_counterfactual_seed448/best.pt \
    ddqn449=runs/ddqn_mc_counterfactual_seed449/best.pt \
    ridge642=runs/ridge_multi_policy_seed642/ridge_domain.joblib \
    ridge643=runs/ridge_multi_policy_seed643/ridge_domain.joblib \
    ridge451=runs/classical_ml_seed541/models/ridge_domain.joblib \
  --opponents rule28-smart,mc,glenn-aloi,couto-ga-stop \
  --games 3000 --seed 20261101 --mc-rollouts 64 \
  --bootstrap-samples 5000 --workers 6 \
  --output-dir results/final_retrain_eval_seed20261101

.venv/bin/python3 scripts/evaluate_ml_head_to_head.py \
  --ppo-models ppo460=runs/ppo_mc_focused_seed460/top_1.pt ppo461=runs/ppo_mc_focused_seed461/top_1.pt \
  --dqn-models ddqn448=runs/ddqn_mc_counterfactual_seed448/best.pt ddqn449=runs/ddqn_mc_counterfactual_seed449/best.pt \
  --models ridge642=runs/ridge_multi_policy_seed642/ridge_domain.joblib ridge643=runs/ridge_multi_policy_seed643/ridge_domain.joblib \
  --games 3000 --seed 20261102 --mc-rollouts 64 \
  --bootstrap-samples 5000 --workers 6 \
  --output-dir results/final_retrain_head_to_head_seed20261102

echo "Final retraining evaluations complete." >> "$LOG"
