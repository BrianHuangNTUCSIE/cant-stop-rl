"""Small supervised neural Stop/Continue policy used by the ML baseline."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

from .model import Move, State
from .agents import Rule28Agent


class SupervisedDNNAgent:
    """Greedy DNN continuation-margin policy with fixed GA pairing."""

    def __init__(self, model: Any, mean: Any, scale: Any, *, device: str = "cpu") -> None:
        self.model = model
        self.mean = mean
        self.scale = scale
        self.device = device
        self.pairing_policy = Rule28Agent(pair_heuristic="ga", smart=True)
        self.model.eval()

    def choose_move(self, state: State, moves: Sequence[Move], rng: random.Random) -> Move:
        return self.pairing_policy.choose_move(state, moves, rng)

    def should_stop(self, state: State, rng: random.Random) -> bool:
        del rng
        import torch
        features = state.features(domain=True)
        x = torch.tensor(
            [(float(value) - float(mu)) / float(scale)
             for value, mu, scale in zip(features, self.mean, self.scale)],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            margin = float(self.model(x).squeeze().item())
        return margin <= 0.0


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> SupervisedDNNAgent:
    import torch
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("algorithm") != "supervised-dnn":
        raise ValueError(f"Not a supervised DNN checkpoint: {path}")
    architecture = payload["architecture"]
    model = torch.nn.Sequential(
        torch.nn.Linear(architecture[0], architecture[1]), torch.nn.ReLU(),
        torch.nn.Linear(architecture[1], architecture[2]), torch.nn.ReLU(),
        torch.nn.Linear(architecture[2], 1),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return SupervisedDNNAgent(model, payload["feature_mean"], payload["feature_scale"], device=device)
