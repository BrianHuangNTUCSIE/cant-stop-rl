#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
SCRIPT="$ROOT/scripts/run_champions_tournament.sh"
PYTHON=".venv/bin/python"

MODE="${1:-full}"
if [ "$MODE" = "--smoke" ]; then
    MODE="smoke"
elif [ "$MODE" = "--worker" ]; then
    MODE="${2:-full}"
elif [ "$MODE" != "full" ]; then
    echo "Usage: $0 [--smoke]"
    exit 2
fi

if [ "$MODE" = "smoke" ]; then
    RUN_ROOT="results/champions_tournament_smoke"
    LOG="results/logs/champions_tournament_smoke.log"
    PID_FILE="results/logs/champions_tournament_smoke.pid"
    GAMES=100
    MC_ROLLOUTS=32
    BOOTSTRAP_SAMPLES=500
else
    RUN_ROOT="results/champions_tournament"
    LOG="results/logs/champions_tournament.log"
    PID_FILE="results/logs/champions_tournament.pid"
    GAMES=5000
    MC_ROLLOUTS=64
    BOOTSTRAP_SAMPLES=5000
fi

DDQN="runs/ddqn_mc_counterfactual_seed447/best.pt"
RIDGE="runs/classical_ml_seed541/models/ridge_domain.joblib"

run_tournament() {
    cleanup() {
        rm -f "$PID_FILE"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    "$PYTHON" scripts/evaluate_ml_head_to_head.py \
        --dqn-models "ddqn-441=$DDQN" \
        --models "ridge-domain=$RIDGE" \
        --agents couto couto-ga-stop glenn-aloi mc \
        --games "$GAMES" \
        --seed 60000641 \
        --mc-rollouts "$MC_ROLLOUTS" \
        --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
        --workers 4 \
        --output-dir "$RUN_ROOT"

    echo "Champions tournament completed."
    echo "Win-rate matrix: $RUN_ROOT/win_rate_matrix.csv"
    echo "Full summary:    $RUN_ROOT/head_to_head_summary.json"
}

if [ "${1:-}" = "--worker" ]; then
    run_tournament
    exit 0
fi

mkdir -p runs

if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Champions tournament is already running (PID $old_pid)."
        echo "Progress: tail -f $LOG"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

if [ -e "$RUN_ROOT" ]; then
    echo "Output already exists: $RUN_ROOT"
    echo "Move or rename it before starting a fresh tournament."
    exit 1
fi

for model in "$DDQN" "$RIDGE"; do
    if [ ! -f "$model" ]; then
        echo "Required model not found: $model"
        exit 1
    fi
done

if [ ! -x "$PYTHON" ]; then
    if ! command -v python3.11 >/dev/null 2>&1; then
        echo "Python 3.11 is required to create .venv (PyTorch does not support the current Python 3.14 setup)."
        exit 1
    fi
    echo "Creating Python 3.11 virtual environment at .venv ..."
    python3.11 -m venv .venv
fi

if ! "$PYTHON" -c "import torch, sklearn, joblib" >/dev/null 2>&1; then
    echo "Installing tournament dependencies into .venv ..."
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install -r requirements-rl.txt -r requirements-ml.txt
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

echo "Started $MODE champions tournament in the background (PID $pid)."
echo "Progress: tail -f $LOG"
echo "Results:  $RUN_ROOT"
