#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

PYTHON=".venv/bin/python"
SCRIPT="$ROOT/scripts/run_ddqn_v2_training.sh"
MODE="full"

terminate_tree() {
    target=$1
    for child in $(pgrep -P "$target" 2>/dev/null || true); do
        terminate_tree "$child"
    done
    kill -TERM "$target" 2>/dev/null || true
}

if [ "${1:-}" = "--stop" ]; then
    found=0
    for pid_file in runs/ddqn_v2_depth_ablation.pid \
            runs/ddqn_v2_depth_ablation_smoke.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                terminate_tree "$pid"
                echo "Stopped DDQN v2 process tree rooted at PID $pid."
                found=1
            fi
            rm -f "$pid_file"
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "No running DDQN v2 experiment found."
    fi
    exit 0
fi

case "${1:-}" in
    "") ;;
    --smoke) MODE="smoke" ;;
    --worker) MODE="${2:-full}" ;;
    *) echo "Usage: $0 [--smoke|--stop]"; exit 2 ;;
esac

if [ "$MODE" = "smoke" ]; then
    RUN_ROOT="runs/ddqn_v2_depth_ablation_smoke"
    LOG="runs/ddqn_v2_depth_ablation_smoke.log"
    PID_FILE="runs/ddqn_v2_depth_ablation_smoke.pid"
else
    RUN_ROOT="runs/ddqn_v2_depth_ablation"
    LOG="runs/ddqn_v2_depth_ablation.log"
    PID_FILE="runs/ddqn_v2_depth_ablation.pid"
fi

run_architecture() {
    name=$1
    hidden_sizes=$2
    seeds=$3
    episodes=$4
    eval_every=$5
    interim_games=$6
    final_games=$7
    collapse_min_episodes=$8

    "$PYTHON" scripts/train_multi_seed.py \
        --seeds "$seeds" \
        --jobs 1 \
        --output-root "$RUN_ROOT/$name" \
        --continue-on-error \
        --algorithm double-dqn \
        --domain-features \
        --network-type standard \
        --pairing-policy ga \
        --reward-shaping 0 \
        --n-step 5 \
        --episodes "$episodes" \
        --num-envs 16 \
        --torch-threads 1 \
        --hidden-sizes "$hidden_sizes" \
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
        --collapse-min-episodes "$collapse_min_episodes" \
        --device cpu
}

run_worker() {
    trap 'rm -f "$PID_FILE"' EXIT
    if [ "$MODE" = "smoke" ]; then
        seeds="941"
        episodes=40
        eval_every=20
        interim_games=20
        final_games=20
        collapse_min_episodes=0
    else
        # Common random numbers across architectures make this a controlled
        # depth-only comparison. Runs are sequential to avoid CPU contention.
        seeds="441,442,443"
        episodes=100000
        eval_every=5000
        interim_games=1000
        final_games=5000
        collapse_min_episodes=15000
    fi

    run_architecture control_128x128 128,128 \
        "$seeds" "$episodes" "$eval_every" "$interim_games" "$final_games" \
        "$collapse_min_episodes"
    run_architecture wide_256x256 256,256 \
        "$seeds" "$episodes" "$eval_every" "$interim_games" "$final_games" \
        "$collapse_min_episodes"
    run_architecture deep_256x256x128 256,256,128 \
        "$seeds" "$episodes" "$eval_every" "$interim_games" "$final_games" \
        "$collapse_min_episodes"
    "$PYTHON" scripts/summarize_depth_ablation.py --root "$RUN_ROOT"
}

if [ "${1:-}" = "--worker" ]; then
    run_worker
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
if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "DDQN v2 depth ablation is already running (PID $old_pid)."
        echo "Progress: tail -f $LOG"
        exit 1
    fi
    rm -f "$PID_FILE"
fi
if [ -e "$RUN_ROOT" ]; then
    echo "Output already exists: $RUN_ROOT"
    echo "Rename it before starting a fresh experiment; existing results are never overwritten."
    exit 1
fi

if command -v caffeinate >/dev/null 2>&1; then
    nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 caffeinate -i "$SCRIPT" --worker "$MODE" > "$LOG" 2>&1 &
else
    nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 "$SCRIPT" --worker "$MODE" > "$LOG" 2>&1 &
fi
pid=$!
echo "$pid" > "$PID_FILE"
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "DDQN v2 failed to start. Last log lines:"
    tail -n 20 "$LOG"
    exit 1
fi
echo "Started $MODE DDQN v2 depth ablation in the background (PID $pid)."
echo "Progress: tail -f $LOG"
echo "Results:  $RUN_ROOT"
