"""PPO actor-critic support for Stop/Continue policies."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .agents import Agent, Rule28Agent
from .model import Move, State
from .rl_env import JointAction, StopContinueAction, StopContinueEnv


class PPOActorCritic(nn.Module):
    """Shared MLP with a configurable policy head and scalar value head."""

    def __init__(self, observation_size: int,
                 hidden_sizes: Sequence[int] = (128, 128),
                 action_size: int = 2) -> None:
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must not be empty")
        if observation_size < 1 or action_size < 1:
            raise ValueError("observation_size and action_size must be positive")
        self.observation_size = observation_size
        self.action_size = action_size
        layers: list[nn.Module] = []
        input_size = observation_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.Tanh()))
            input_size = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_size, action_size)
        self.value_head = nn.Linear(input_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observations)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


class PPOAgent:
    """Evaluation-time Agent wrapper around a trained PPO policy."""

    def __init__(self, model: PPOActorCritic, *, pairing_policy: Agent | None = None,
                 device: torch.device | str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.pairing_policy = pairing_policy or Rule28Agent(pair_heuristic="ga", smart=True)
        self.learn_pairing = model.action_size == 8 and pairing_policy is None
        if model.action_size not in (2, 8):
            raise ValueError("PPO evaluation supports 2 or 8 actions")
        self.strategic_features = bool(getattr(self.model, "strategic_features", False))
        expected = (80 if self.strategic_features else 68) + (18 if self.learn_pairing else 0)
        if model.observation_size != expected:
            raise ValueError(
                f"PPO policy mode expects {expected} observations, got "
                f"{model.observation_size}"
            )

    def choose_move(self, state: State, moves: Sequence[Move], rng: random.Random) -> Move:
        if self.learn_pairing:
            ordered = sorted(moves)
            features = state.features(
                strategic=self.strategic_features,
                domain=not self.strategic_features,
            ) + StopContinueEnv._move_features(ordered)
            with torch.inference_mode():
                logits, _ = self.model(torch.tensor(
                    [features], dtype=torch.float32, device=self.device
                ))
            return ordered[int(logits[0, :len(ordered)].argmax().item())]
        return self.pairing_policy.choose_move(state, moves, rng)

    def should_stop(self, state: State, rng: random.Random) -> bool:
        del rng
        features = state.features(
            strategic=self.strategic_features,
            domain=not self.strategic_features,
        )
        if self.learn_pairing:
            features += [0.0] * 18
        with torch.inference_mode():
            logits, _ = self.model(torch.tensor([features], dtype=torch.float32, device=self.device))
        if self.learn_pairing:
            action = int(logits[:, 6:].argmax(dim=-1).item()) + 6
            return action == int(JointAction.STOP)
        return int(logits.argmax(dim=-1).item()) == int(StopContinueAction.STOP)


def mask_policy_logits(logits: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    """Exclude illegal actions from a categorical PPO policy."""
    if logits.shape != legal_masks.shape:
        raise ValueError("legal masks must have the same shape as policy logits")
    if legal_masks.dtype is not torch.bool:
        legal_masks = legal_masks.bool()
    if not bool(legal_masks.any(dim=1).all()):
        raise ValueError("every policy observation must have a legal action")
    return logits.masked_fill(~legal_masks, float("-inf"))


def joint_action_masks(observations: torch.Tensor) -> torch.Tensor:
    """Infer pair/decision legal masks from the 18 candidate features."""
    if observations.ndim != 2 or observations.shape[1] != 86:
        raise ValueError("joint PPO observations must have shape (batch, 86)")
    valid = observations[:, 70:86:3] > 0.5
    pair_phase = valid.any(dim=1)
    masks = torch.zeros(
        (observations.shape[0], 8), dtype=torch.bool,
        device=observations.device,
    )
    masks[:, :6] = valid
    masks[:, 6] = ~pair_phase
    masks[:, 7] = ~pair_phase
    return masks


def initialize_from_ddqn(model: PPOActorCritic, checkpoint: str | Path,
                          device: torch.device | str = "cpu") -> None:
    """Copy compatible DDQN trunk and Q logits into a PPO model.

    The DDQN uses ReLU while PPO uses Tanh, so only linear weights and biases
    are copied.  This gives PPO the strong model's representation and initial
    action preference without pretending that the DDQN output is a value head.
    """
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload.get("config", {})
    if config.get("observation_size") != model.trunk[0].in_features:
        raise ValueError("DDQN observation size does not match PPO model")
    if tuple(config.get("hidden_sizes", ())) != tuple(
        layer.out_features for layer in model.trunk if isinstance(layer, nn.Linear)
    ):
        raise ValueError("DDQN hidden sizes do not match PPO model")
    state = payload.get("online_state_dict")
    if state is None:
        raise ValueError("checkpoint has no online_state_dict")
    with torch.no_grad():
        ppo_linear = [layer for layer in model.trunk if isinstance(layer, nn.Linear)]
        for layer_index, destination in enumerate(ppo_linear):
            source_index = 2 * layer_index
            destination.weight.copy_(state[f"network.{source_index}.weight"])
            destination.bias.copy_(state[f"network.{source_index}.bias"])
        model.policy_head.weight.copy_(state["network.4.weight"])
        model.policy_head.bias.copy_(state["network.4.bias"])


def initialize_from_ppo(model: PPOActorCritic, checkpoint: str | Path,
                        device: torch.device | str = "cpu") -> dict[str, object]:
    """Expand a fixed-pairing PPO into an 86-input, eight-action joint policy."""
    source, config = load_checkpoint(checkpoint, device=device)
    if source.action_size != 2 or source.observation_size != 68:
        raise ValueError("initial PPO must be a 68-input, two-action checkpoint")
    if model.action_size not in (2, 8) or model.observation_size not in (80, 86):
        raise ValueError("destination PPO must be a strategic fixed-pairing model")
    source_linears = [layer for layer in source.trunk if isinstance(layer, nn.Linear)]
    destination_linears = [layer for layer in model.trunk if isinstance(layer, nn.Linear)]
    if [layer.out_features for layer in source_linears] != [
            layer.out_features for layer in destination_linears]:
        raise ValueError("initial PPO hidden sizes do not match destination model")
    with torch.no_grad():
        first_source, first_destination = source_linears[0], destination_linears[0]
        first_destination.weight.zero_()
        first_destination.weight[:, :68].copy_(first_source.weight)
        first_destination.bias.copy_(first_source.bias)
        for source_layer, destination_layer in zip(
                source_linears[1:], destination_linears[1:]):
            destination_layer.weight.copy_(source_layer.weight)
            destination_layer.bias.copy_(source_layer.bias)
        model.value_head.load_state_dict(source.value_head.state_dict())
        if model.action_size == 8:
            model.policy_head.weight[6:8].copy_(source.policy_head.weight)
            model.policy_head.bias[6:8].copy_(source.policy_head.bias)
        else:
            model.policy_head.load_state_dict(source.policy_head.state_dict())
    model.strategic_features = model.observation_size in (80, 86)
    return config


def save_checkpoint(path: str | Path, model: PPOActorCritic, optimizer: torch.optim.Optimizer,
                    config: dict[str, object], *, episodes: int, updates: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_version": 1,
        "algorithm": "ppo",
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episodes": episodes,
        "updates": updates,
    }, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[PPOActorCritic, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = dict(payload["config"])
    model = PPOActorCritic(
        config["observation_size"], tuple(config["hidden_sizes"]),
        int(config.get("action_size", 2)),
    )
    model.strategic_features = bool(config.get("strategic_features", False))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device), config
