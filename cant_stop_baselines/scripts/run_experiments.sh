#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

RUN_ROOT="runs/ddqn_robust_5seeds"
LOG="results/logs/ddqn_robust_5seeds.log"
PID_FILE="results/logs/ddqn_robust_5seeds.pid"

mkdir -p runs

if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Experiments are already running (PID $old_pid)."
        echo "Progress: tail -f $LOG"
        exit 1
    fi
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Missing .venv/bin/python; create the virtual environment first."
    exit 1
fi

if [ -e "$RUN_ROOT" ]; then
    echo "Output already exists: $RUN_ROOT"
    echo "Move or remove it before starting a fresh experiment."
    exit 1
fi

nohup env PYTHONUNBUFFERED=1 caffeinate -i .venv/bin/python scripts/train_multi_seed.py \
    --seeds 441,442,443,444,445 \
    --jobs 1 \
    --output-root "$RUN_ROOT" \
    --continue-on-error \
    --algorithm double-dqn \
    --domain-features \
    --n-step 5 \
    --episodes 100000 \
    --num-envs 16 \
    --torch-threads 1 \
    --hidden-sizes 128,128 \
    --epsilon-decay-steps 1500000 \
    --epsilon-end 0.03 \
    --replay-capacity 500000 \
    --opponent-mix rule28-smart:0.35,mc:0.35,rule28-ga-pure:0.1,rule28:0.1,random:0.1 \
    --training-mc-rollouts 16 \
    --validation-opponents rule28-smart,mc \
    --validation-mc-rollouts 32 \
    --eval-every 5000 \
    --interim-eval-games 1000 \
    --eval-games 5000 \
    --checkpoint-every 5000 \
    --device cpu \
    > "$LOG" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"

echo "Started all experiments in the background (PID $pid)."
echo "Progress: tail -f $LOG"
echo "Results:  $RUN_ROOT"
