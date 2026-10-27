#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
SCRIPT="$ROOT/scripts/run_classical_ml_experiments.sh"

RUN_ROOT="runs/classical_ml_seed541"
LOG="results/logs/classical_ml_seed541.log"
PID_FILE="results/logs/classical_ml_seed541.pid"
PYTHON=".venv/bin/python"

run_pipeline() {
    cleanup() {
        rm -f "$PID_FILE"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    echo "[1/3] Generating paired-rollout dataset"
    "$PYTHON" scripts/generate_ml_dataset.py \
        --games 1000 \
        --max-states 10000 \
        --sample-probability 0.5 \
        --exploration-probability 0.5 \
        --rollouts 32 \
        --seed 541 \
        --output "$RUN_ROOT/dataset.csv"

    echo "[2/3] Training classical ML models"
    "$PYTHON" scripts/train_ml_models.py \
        --dataset "$RUN_ROOT/dataset.csv" \
        --models all \
        --features all \
        --seed 641 \
        --output-dir "$RUN_ROOT/models"

    echo "[3/3] Evaluating domain-feature models"
    "$PYTHON" scripts/evaluate_ml.py \
        --models \
            "$RUN_ROOT/models/logistic_domain.joblib" \
            "$RUN_ROOT/models/decision-tree_domain.joblib" \
            "$RUN_ROOT/models/random-forest_domain.joblib" \
            "$RUN_ROOT/models/linear-svm_domain.joblib" \
            "$RUN_ROOT/models/ridge_domain.joblib" \
        --opponents rule28-smart,mc \
        --games 5000 \
        --seed 40000641 \
        --mc-rollouts 32 \
        --bootstrap-samples 5000 \
        --output-dir "$RUN_ROOT/evaluation"

    echo "Completed classical ML experiments."
    echo "Training report: $RUN_ROOT/models/training_report.json"
    echo "Evaluation report: $RUN_ROOT/evaluation/evaluation_summary.json"
}

if [ "${1:-}" = "--worker" ]; then
    run_pipeline
    exit 0
fi

mkdir -p runs

if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Classical ML experiments are already running (PID $old_pid)."
        echo "Progress: tail -f $LOG"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

if [ ! -x "$PYTHON" ]; then
    echo "Missing $PYTHON; create the virtual environment first."
    exit 1
fi

if ! "$PYTHON" -c "import sklearn, joblib" >/dev/null 2>&1; then
    echo "Missing classical ML dependencies. Install them with:"
    echo "  $PYTHON -m pip install -r requirements-ml.txt"
    exit 1
fi

if [ -e "$RUN_ROOT" ]; then
    echo "Output already exists: $RUN_ROOT"
    echo "Move or remove it before starting a fresh experiment."
    exit 1
fi

nohup env PYTHONUNBUFFERED=1 caffeinate -i "$SCRIPT" --worker > "$LOG" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"

echo "Started classical ML experiments in the background (PID $pid)."
echo "Progress: tail -f $LOG"
echo "Results:  $RUN_ROOT"
