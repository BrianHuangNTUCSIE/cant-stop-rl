#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

PYTHON=".venv/bin/python"
SCRIPT="$ROOT/scripts/run_ddqn_mc_curriculum.sh"
RUN_DIR="runs/ddqn_mc_curriculum_seed441"
LOG="runs/ddqn_mc_curriculum_seed441.log"
PID_FILE="runs/ddqn_mc_curriculum_seed441.pid"

terminate_tree() {
    target=$1
    children=$(pgrep -P "$target" 2>/dev/null || true)
    for child in $children; do
        terminate_tree "$child"
    done
    kill -TERM "$target" 2>/dev/null || true
}

if [ "${1:-}" = "--stop" ]; then
    if [ ! -f "$PID_FILE" ]; then
        echo "No running MC-heavy DDQN experiment found."
        exit 0
    fi
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        terminate_tree "$pid"
        echo "Stopped MC-heavy DDQN process tree rooted at PID $pid."
    else
        echo "No running MC-heavy DDQN experiment found."
    fi
    rm -f "$PID_FILE"
    exit 0
fi

MODE="${1:-full}"
case "$MODE" in
    full|--worker-full) ;;
    --smoke|--worker-smoke) ;;
    *) echo "Usage: $0 [--smoke|--stop]"; exit 2 ;;
esac

run_training() {
    mode=$1
    trap 'rm -f "$PID_FILE"' EXIT
    if [ "$mode" = "smoke" ]; then
        episodes=40
        eval_every=20
        interim_games=20
        final_games=20
        output_dir="${RUN_DIR}_smoke"
    else
        episodes=100000
        eval_every=5000
        interim_games=1000
        final_games=5000
        output_dir="$RUN_DIR"
    fi

    "$PYTHON" scripts/train_dqn.py \
        --algorithm double-dqn \
        --domain-features \
        --network-type standard \
        --pairing-policy ga \
        --reward-shaping 0 \
        --n-step 5 \
        --episodes "$episodes" \
        --num-envs 16 \
        --torch-threads 1 \
        --seed 441 \
        --output-dir "$output_dir" \
        --hidden-sizes 128,128 \
        --learning-rate 0.0003 \
        --batch-size 256 \
        --replay-capacity 500000 \
        --warmup-steps 10000 \
        --train-frequency 4 \
        --target-update-steps 5000 \
        --epsilon-start 1.0 \
        --epsilon-decay-steps 1500000 \
        --epsilon-end 0.03 \
        --gradient-clip 5.0 \
        --opponent-mix rule28-smart:0.35,mc:0.35,rule28-ga-pure:0.1,rule28:0.1,random:0.1 \
        --mc-heavy-curriculum \
        --training-mc-rollouts 16 \
        --validation-opponents rule28-smart,mc \
        --validation-mc-rollouts 32 \
        --validation-seed-blocks 1 \
        --checkpoint-metric minimum-win-rate \
        --eval-every "$eval_every" \
        --interim-eval-games "$interim_games" \
        --eval-games "$final_games" \
        --checkpoint-every "$eval_every" \
        --collapse-bust-threshold 0.55 \
        --collapse-patience 2 \
        --collapse-min-episodes 15000 \
        --device cpu
}

if [ "$MODE" = "--worker-full" ]; then
    run_training full
    exit 0
fi
if [ "$MODE" = "--worker-smoke" ]; then
    run_training smoke
    exit 0
fi

mkdir -p runs
if [ ! -x "$PYTHON" ]; then
    echo "Missing $PYTHON; create the virtual environment and install requirements-rl.txt."
    exit 1
fi
if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
    echo "PyTorch is missing from $PYTHON; install requirements-rl.txt."
    exit 1
fi

if [ "$MODE" = "--smoke" ]; then
    worker_mode="--worker-smoke"
    output_dir="${RUN_DIR}_smoke"
    run_log="${LOG%.log}_smoke.log"
else
    worker_mode="--worker-full"
    output_dir="$RUN_DIR"
    run_log="$LOG"
fi

if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "MC-heavy DDQN experiment is already running (PID $old_pid)."
        exit 1
    fi
    rm -f "$PID_FILE"
fi
if [ -e "$output_dir" ]; then
    echo "Output already exists: $output_dir"
    echo "Rename it before starting a fresh experiment; existing results are never overwritten."
    exit 1
fi

if command -v caffeinate >/dev/null 2>&1; then
    nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 caffeinate -i "$SCRIPT" "$worker_mode" > "$run_log" 2>&1 &
else
    nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 "$SCRIPT" "$worker_mode" > "$run_log" 2>&1 &
fi
pid=$!
echo "$pid" > "$PID_FILE"
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "MC-heavy DDQN experiment failed to start. Last log lines:"
    tail -n 20 "$run_log"
    exit 1
fi

echo "Started MC-heavy DDQN experiment in the background (PID $pid)."
echo "Progress: tail -f $run_log"
echo "Results:  $output_dir"
