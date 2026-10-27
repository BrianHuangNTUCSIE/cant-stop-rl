"""Train a DNN on the same counterfactual dataset used by classical ML."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from torch import nn

from train_ml_models import grouped_split


def load_data(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.asarray([[float(row[f"x{i:02d}"]) for i in range(68)] for row in rows], dtype=np.float32)
    margins = np.asarray([float(row["continue_margin"]) for row in rows], dtype=np.float32)
    weights = np.asarray([max(abs(float(row["continue_margin"])), 1.0 / int(row["rollouts"])) for row in rows], dtype=np.float32)
    return rows, x, margins, weights


def train_model(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                train: np.ndarray, validation: np.ndarray, *, seed: int,
                hidden: int, epochs: int, lr: float) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(68, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    tx = torch.tensor(x[train]); ty = torch.tensor(y[train]).unsqueeze(1)
    tw = torch.tensor(weights[train]).unsqueeze(1)
    vx = torch.tensor(x[validation]); vy = torch.tensor(y[validation])
    mean = x[train].mean(axis=0); scale = x[train].std(axis=0); scale[scale < 1e-6] = 1.0
    tx = (tx - torch.tensor(mean)) / torch.tensor(scale)
    vx = (vx - torch.tensor(mean)) / torch.tensor(scale)
    best_state = None; best_loss = float("inf")
    for _ in range(epochs):
        model.train(); optimizer.zero_grad()
        prediction = model(tx)
        loss = (((prediction - ty) ** 2) * tw).mean()
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(((model(vx).squeeze(1) - vy) ** 2).mean().item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    with torch.no_grad():
        predicted = model(vx).squeeze(1).numpy()
    actions = np.where(predicted > 0, 0, 1)
    labels = np.where(y[validation] > 0, 0, 1)
    metric = {
        "mse": float(np.mean((predicted - y[validation]) ** 2)),
        "action_accuracy": float(np.mean(actions == labels)),
        "mean_rollout_regret": float(np.mean(np.abs(y[validation]) * (actions != labels))),
    }
    model._feature_mean = mean  # type: ignore[attr-defined]
    model._feature_scale = scale  # type: ignore[attr-defined]
    return model, metric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(1)
    rows, x, margins, weights = load_data(args.dataset)
    splits = grouped_split(rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    candidates = [(64, 1e-3), (128, 1e-3), (128, 3e-4)]
    best = None
    for hidden, lr in candidates:
        model, validation = train_model(x, margins, weights, splits["train"], splits["validation"],
                                        seed=args.seed + hidden, hidden=hidden, epochs=args.epochs, lr=lr)
        ranking = (validation["mean_rollout_regret"], -validation["action_accuracy"])
        if best is None or ranking < best[0]:
            best = (ranking, hidden, lr, model, validation)
    assert best is not None
    _, hidden, lr, _, validation = best
    final_indices = np.concatenate((splits["train"], splits["validation"]))
    model, _ = train_model(x, margins, weights, final_indices, splits["validation"],
                           seed=args.seed + hidden + 1000, hidden=hidden, epochs=args.epochs, lr=lr)
    checkpoint = args.output_dir / "supervised_dnn_domain.pt"
    torch.save({"algorithm": "supervised-dnn", "architecture": [68, hidden, hidden],
                "model_state_dict": model.state_dict(),
                "feature_mean": model._feature_mean.tolist(), "feature_scale": model._feature_scale.tolist(),
                "config": {"dataset": str(args.dataset), "seed": args.seed, "hidden": hidden,
                           "learning_rate": lr, "epochs": args.epochs, "target": "continue_margin",
                           "split": "70/15/15_grouped_by_source_game"}}, checkpoint)
    report = {"dataset": str(args.dataset), "seed": args.seed, "selected_hidden": hidden,
              "learning_rate": lr, "epochs": args.epochs, "validation_selection": validation,
              "split_states": {key: int(len(value)) for key, value in splits.items()},
              "checkpoint": str(checkpoint), "elapsed_seconds": time.perf_counter() - started}
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
