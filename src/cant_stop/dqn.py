"""PyTorch components for DQN and Double DQN Stop/Continue policies.

PyTorch is deliberately isolated in this module so the baseline simulator keeps
working without third-party dependencies.  Install ``requirements-rl.txt`` to
train or load a neural policy.
"""
from __future__ import annotations

import inspect
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "DQN support requires PyTorch. Install requirements-rl.txt with a "
        "PyTorch-supported Python version."
    ) from exc

from .agents import Rule28Agent
from .model import Move, State
from .rl_env import JointAction, StopContinueAction


CHECKPOINT_VERSION = 1


def available_device_types() -> tuple[str, ...]:
    """Return accelerator types usable by this PyTorch installation."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("mps")
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        devices.append("xpu")
    return tuple(devices)


def select_device(requested: str = "auto") -> torch.device:
    """Choose CUDA, Apple MPS, Intel XPU, or CPU without platform assumptions."""
    requested = requested.lower()
    supported = {"auto", "cpu", "cuda", "mps", "xpu"}
    if requested not in supported:
        raise ValueError(f"Unknown device {requested!r}; choose from {sorted(supported)}")
    available = available_device_types()
    if requested == "auto":
        for candidate in ("cuda", "mps", "xpu", "cpu"):
            if candidate in available:
                return torch.device(candidate)
    elif requested in available:
        return torch.device(requested)
    raise RuntimeError(
        f"Requested device {requested!r} is unavailable; available: {', '.join(available)}"
    )


def seed_torch(seed: int, deterministic: bool = True) -> None:
    """Seed PyTorch backends and request deterministic kernels when possible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            cudnn.benchmark = False
            cudnn.deterministic = True


@dataclass(frozen=True)
class DQNConfig:
    observation_size: int = 58
    action_size: int = 2
    hidden_sizes: tuple[int, ...] = (128, 128)
    algorithm: str = "dqn"
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 250_000
    warmup_steps: int = 10_000
    train_frequency: int = 4
    target_update_steps: int = 5_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 300_000
    gradient_clip: float = 5.0
    n_step: int = 1
    network_type: str = "standard"
    pairing_policy: str = "ga"
    replay_type: str = "uniform"
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    counterfactual_loss_weight: float = 0.0
    counterfactual_target_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.observation_size < 1 or self.action_size < 1:
            raise ValueError("observation_size and action_size must be positive")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if self.algorithm not in ("dqn", "double-dqn"):
            raise ValueError("algorithm must be 'dqn' or 'double-dqn'")
        if self.network_type not in ("standard", "dueling"):
            raise ValueError("network_type must be 'standard' or 'dueling'")
        if self.pairing_policy not in ("ga", "couto", "learned"):
            raise ValueError("pairing_policy must be 'ga', 'couto', or 'learned'")
        if self.replay_type not in ("uniform", "prioritized"):
            raise ValueError("replay_type must be 'uniform' or 'prioritized'")
        if self.per_alpha < 0.0 or self.per_beta_start < 0.0 or self.per_beta_end > 1.0:
            raise ValueError("invalid prioritized replay parameters")
        if self.per_beta_start > self.per_beta_end:
            raise ValueError("per_beta_start must not exceed per_beta_end")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        for name in ("learning_rate", "batch_size", "replay_capacity",
                     "train_frequency", "target_update_steps",
                     "epsilon_decay_steps", "gradient_clip"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.n_step < 1:
            raise ValueError("n_step must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.counterfactual_loss_weight < 0.0:
            raise ValueError("counterfactual_loss_weight must be non-negative")
        if self.counterfactual_target_scale <= 0.0:
            raise ValueError("counterfactual_target_scale must be positive")


class QNetwork(nn.Module):
    """Small multilayer perceptron mapping a state to action values."""

    def __init__(self, observation_size: int, action_size: int,
                 hidden_sizes: Sequence[int] = (128, 128),
                 network_type: str = "standard") -> None:
        super().__init__()
        if network_type not in ("standard", "dueling"):
            raise ValueError("network_type must be 'standard' or 'dueling'")
        self.action_size = action_size
        self.network_type = network_type
        output_size = action_size if network_type == "standard" else hidden_sizes[-1]
        trunk_sizes = ([observation_size, *hidden_sizes, output_size]
                       if network_type == "standard"
                       else [observation_size, *hidden_sizes])
        layers: list[nn.Module] = []
        for index, (input_size, layer_size) in enumerate(
                zip(trunk_sizes, trunk_sizes[1:])):
            layers.append(nn.Linear(input_size, layer_size))
            if index < len(trunk_sizes) - 2 or network_type == "dueling":
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        if network_type == "dueling":
            self.value_head = nn.Linear(hidden_sizes[-1], 1)
            self.advantage_head = nn.Linear(hidden_sizes[-1], action_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        values = self.network(observations)
        if self.network_type == "standard":
            return values
        value = self.value_head(values)
        advantage = self.advantage_head(values)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """Preallocated CPU tensor ring buffer with reproducible sampling.

    Sampling indices directly avoids copying the entire replay memory on every
    gradient update, which was the main CPU bottleneck in the initial trainer.
    """

    def __init__(self, capacity: int, observation_size: int,
                 rng: random.Random | None = None) -> None:
        if capacity < 1 or observation_size < 1:
            raise ValueError("capacity and observation_size must be positive")
        self.capacity = capacity
        self.observation_size = observation_size
        self._rng = rng or random.Random()
        self._states = torch.empty((capacity, observation_size), dtype=torch.float32)
        self._actions = torch.empty(capacity, dtype=torch.long)
        self._rewards = torch.empty(capacity, dtype=torch.float32)
        self._next_states = torch.empty(
            (capacity, observation_size), dtype=torch.float32
        )
        self._dones = torch.empty(capacity, dtype=torch.bool)
        self._n_steps = torch.empty(capacity, dtype=torch.long)
        self._counterfactual = torch.zeros((capacity, 2), dtype=torch.float32)
        self._counterfactual_mask = torch.zeros(capacity, dtype=torch.bool)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, state: Sequence[float], action: int, reward: float,
             next_state: Sequence[float] | None, done: bool,
             n_steps: int = 1, counterfactual: Sequence[float] | None = None,
             counterfactual_mask: bool = False) -> None:
        self.push_batch([state], [action], [reward], [next_state], [done], [n_steps],
                        None if counterfactual is None else [counterfactual],
                        [counterfactual_mask])

    def push_batch(self, states: Sequence[Sequence[float]], actions: Sequence[int],
                   rewards: Sequence[float],
                   next_states: Sequence[Sequence[float] | None],
                   dones: Sequence[bool],
                   n_steps: Sequence[int] | None = None,
                   counterfactuals: Sequence[Sequence[float]] | None = None,
                   counterfactual_masks: Sequence[bool] | None = None) -> None:
        """Insert a collection of transitions with one tensor conversion."""
        count = len(states)
        if n_steps is None:
            n_steps = [1] * count
        if counterfactuals is None:
            counterfactuals = [[0.0, 0.0]] * count
        if counterfactual_masks is None:
            counterfactual_masks = [False] * count
        if not (count == len(actions) == len(rewards)
                == len(next_states) == len(dones) == len(n_steps)
                == len(counterfactuals) == len(counterfactual_masks)):
            raise ValueError("all transition fields must have the same length")
        if any(value < 1 for value in n_steps):
            raise ValueError("n_steps must be positive")
        if count == 0:
            return
        if count > self.capacity:
            start = count - self.capacity
            states, actions, rewards = states[start:], actions[start:], rewards[start:]
            next_states, dones, n_steps = (
                next_states[start:], dones[start:], n_steps[start:]
            )
            counterfactuals = counterfactuals[start:]
            counterfactual_masks = counterfactual_masks[start:]
            count = self.capacity
        if any(len(state) != self.observation_size for state in states):
            raise ValueError("state has the wrong observation size")
        if any(state is not None and len(state) != self.observation_size
               for state in next_states):
            raise ValueError("next_state has the wrong observation size")

        state_tensor = torch.tensor(states, dtype=torch.float32)
        next_state_tensor = torch.tensor([
            ([0.0] * self.observation_size if state is None else state)
            for state in next_states
        ], dtype=torch.float32)
        indices = torch.tensor(
            [(self._position + offset) % self.capacity for offset in range(count)],
            dtype=torch.long,
        )
        self._states.index_copy_(0, indices, state_tensor)
        self._actions.index_copy_(
            0, indices, torch.tensor(actions, dtype=torch.long)
        )
        self._rewards.index_copy_(
            0, indices, torch.tensor(rewards, dtype=torch.float32)
        )
        self._next_states.index_copy_(0, indices, next_state_tensor)
        self._dones.index_copy_(
            0, indices, torch.tensor(dones, dtype=torch.bool)
        )
        self._n_steps.index_copy_(
            0, indices, torch.tensor(n_steps, dtype=torch.long)
        )
        self._counterfactual.index_copy_(
            0, indices, torch.tensor(counterfactuals, dtype=torch.float32)
        )
        self._counterfactual_mask.index_copy_(
            0, indices, torch.tensor(counterfactual_masks, dtype=torch.bool)
        )
        self._position = (self._position + count) % self.capacity
        self._size = min(self.capacity, self._size + count)

    def push_transition_batch(self, batch) -> None:
        """Insert a backend-neutral ``TransitionBatch`` without unpacking it at call sites."""
        if batch.observation_size != self.observation_size:
            raise ValueError("transition batch has the wrong observation size")
        self.push_batch(*batch.replay_fields())

    def sample(self, batch_size: int, device: torch.device,
               *, include_n_steps: bool = False,
               include_priorities: bool = False,
               include_counterfactual: bool = False) -> tuple[torch.Tensor, ...]:
        if include_priorities:
            raise ValueError("uniform replay does not provide priorities")
        if batch_size < 1 or batch_size > self._size:
            raise ValueError("batch_size must be between 1 and the buffer length")
        indices = torch.tensor(
            self._rng.sample(range(self._size), batch_size), dtype=torch.long
        )
        values = tuple(tensor.index_select(0, indices).to(device) for tensor in (
            self._states, self._actions, self._rewards,
            self._next_states, self._dones,
        ))
        if include_n_steps:
            values += (self._n_steps.index_select(0, indices).to(device),)
        if include_counterfactual:
            values += (
                self._counterfactual.index_select(0, indices).to(device),
                self._counterfactual_mask.index_select(0, indices).to(device),
            )
        return values


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized replay with importance-sampling correction."""

    def __init__(self, capacity: int, observation_size: int,
                 rng: random.Random | None = None, *, alpha: float = 0.6,
                 beta: float = 0.4, priority_epsilon: float = 1e-5) -> None:
        super().__init__(capacity, observation_size, rng)
        if alpha < 0.0 or beta < 0.0 or beta > 1.0 or priority_epsilon <= 0.0:
            raise ValueError("invalid prioritized replay parameters")
        self.alpha = alpha
        self.beta = beta
        self.priority_epsilon = priority_epsilon
        self._priorities = torch.zeros(capacity, dtype=torch.float32)
        self._max_priority = 1.0

    def set_beta(self, beta: float) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.beta = beta

    def push_batch(self, states, actions, rewards, next_states, dones,
                   n_steps=None, counterfactuals=None,
                   counterfactual_masks=None) -> None:
        count = len(states)
        if count == 0:
            return
        actual = min(count, self.capacity)
        start = max(0, count - self.capacity)
        # ReplayBuffer truncates oversized batches before writing, then starts
        # at the current ring position.
        first_index = self._position
        super().push_batch(states, actions, rewards, next_states, dones, n_steps,
                           counterfactuals, counterfactual_masks)
        indices = [(first_index + offset) % self.capacity
                   for offset in range(actual)]
        self._priorities[torch.tensor(indices, dtype=torch.long)] = self._max_priority

    def sample(self, batch_size: int, device: torch.device,
               *, include_n_steps: bool = False,
               include_priorities: bool = False,
               include_counterfactual: bool = False) -> tuple[torch.Tensor, ...]:
        if not include_priorities:
            return super().sample(
                batch_size, device, include_n_steps=include_n_steps,
                include_counterfactual=include_counterfactual,
            )
        if batch_size < 1 or batch_size > self._size:
            raise ValueError("batch_size must be between 1 and the buffer length")
        priorities = self._priorities[:self._size].clamp_min(self.priority_epsilon)
        probabilities = priorities.pow(self.alpha)
        probabilities /= probabilities.sum()
        indices = torch.multinomial(probabilities, batch_size, replacement=False)
        values = tuple(tensor.index_select(0, indices).to(device) for tensor in (
            self._states, self._actions, self._rewards,
            self._next_states, self._dones,
        ))
        if include_n_steps:
            values += (self._n_steps.index_select(0, indices).to(device),)
        if include_counterfactual:
            values += (
                self._counterfactual.index_select(0, indices).to(device),
                self._counterfactual_mask.index_select(0, indices).to(device),
            )
        weights = (self._size * probabilities.index_select(0, indices)).pow(-self.beta)
        weights /= weights.max().clamp_min(1e-12)
        return values + (weights.to(device), indices.to(device))

    def update_priorities(self, indices: torch.Tensor,
                          priorities: torch.Tensor) -> None:
        cpu_indices = indices.detach().cpu().long()
        values = priorities.detach().cpu().float().abs().clamp_min(self.priority_epsilon)
        self._priorities.index_copy_(0, cpu_indices, values)
        self._max_priority = max(self._max_priority, float(values.max().item()))


@dataclass(frozen=True)
class NStepTransition:
    state: Sequence[float]
    action: int
    reward: float
    next_state: Sequence[float] | None
    done: bool
    n_steps: int
    counterfactual: Sequence[float]
    counterfactual_mask: bool


class NStepAccumulator:
    """Convert one episode's one-step transitions into discounted n-step ones."""

    def __init__(self, n_step: int, gamma: float) -> None:
        if n_step < 1:
            raise ValueError("n_step must be positive")
        self.n_step = n_step
        self.gamma = gamma
        self._queue: deque[tuple[Sequence[float], int, float,
                                 Sequence[float] | None, bool,
                                 Sequence[float], bool]] = deque()

    def append(self, state: Sequence[float], action: int, reward: float,
               next_state: Sequence[float] | None,
               done: bool, counterfactual: Sequence[float] | None = None,
               counterfactual_mask: bool = False) -> list[NStepTransition]:
        self._queue.append((state, action, reward, next_state, done,
                            counterfactual or (0.0, 0.0), counterfactual_mask))
        emitted: list[NStepTransition] = []
        if done:
            while self._queue:
                emitted.append(self._emit_oldest())
        elif len(self._queue) >= self.n_step:
            emitted.append(self._emit_oldest())
        return emitted

    def _emit_oldest(self) -> NStepTransition:
        horizon = min(self.n_step, len(self._queue))
        total_reward = 0.0
        final_next_state: Sequence[float] | None = None
        final_done = False
        actual_steps = 0
        for index, (_, _, reward, next_state, done, _, _) in enumerate(self._queue):
            if index >= horizon:
                break
            total_reward += (self.gamma ** index) * reward
            final_next_state = next_state
            final_done = done
            actual_steps = index + 1
            if done:
                break
        state, action, _, _, _, counterfactual, counterfactual_mask = self._queue.popleft()
        return NStepTransition(
            state, action, total_reward, final_next_state, final_done, actual_steps,
            counterfactual, counterfactual_mask,
        )


def epsilon_at_step(config: DQNConfig, step: int) -> float:
    fraction = min(max(step, 0) / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def choose_action(model: QNetwork, observation: Sequence[float], epsilon: float,
                  rng: random.Random, device: torch.device) -> int:
    return choose_actions(model, [observation], epsilon, rng, device)[0]


def choose_actions(model: QNetwork, observations: Sequence[Sequence[float]],
                   epsilon: float, rng: random.Random,
                   device: torch.device,
                   legal_masks: Sequence[Sequence[bool]] | None = None) -> list[int]:
    """Choose epsilon-greedy actions with one network call for greedy states."""
    action_size = model.action_size
    if legal_masks is not None and len(legal_masks) != len(observations):
        raise ValueError("legal_masks must match observations")
    actions = [-1] * len(observations)
    greedy_indices: list[int] = []
    for index in range(len(observations)):
        if rng.random() < epsilon:
            choices = ([i for i, legal in enumerate(legal_masks[index]) if legal]
                       if legal_masks is not None else list(range(action_size)))
            if not choices:
                raise ValueError("an observation has no legal actions")
            actions[index] = rng.choice(choices)
        else:
            greedy_indices.append(index)
    if greedy_indices:
        with torch.no_grad():
            batch = torch.tensor(
                [observations[index] for index in greedy_indices],
                dtype=torch.float32, device=device,
            )
            values = model(batch)
            if legal_masks is not None:
                for row, index in enumerate(greedy_indices):
                    illegal = torch.tensor(
                        [not value for value in legal_masks[index]],
                        dtype=torch.bool, device=device,
                    )
                    values[row, illegal] = float("-inf")
            greedy_actions = values.argmax(dim=1).cpu().tolist()
        for index, action in zip(greedy_indices, greedy_actions):
            actions[index] = int(action)
    return actions


def joint_legal_action_mask(observations: torch.Tensor) -> torch.Tensor:
    """Infer legal joint actions from the candidate-pair feature slots.

    In learned-pairing mode a non-empty candidate list identifies the pair
    phase; otherwise the observation is at the Stop/Continue decision.  This
    is used for target bootstrapping as well as behaviour-policy selection so
    illegal actions cannot inflate Double-DQN targets.
    """
    if observations.ndim != 2 or observations.shape[1] < 18:
        raise ValueError("joint observations must be a two-dimensional tensor")
    base = observations.shape[1] - 18
    valid = observations[:, base + 2:base + 18:3] > 0.5
    pair_phase = valid.any(dim=1)
    masks = torch.zeros(
        (observations.shape[0], 8), dtype=torch.bool, device=observations.device
    )
    masks[:, 6] = ~pair_phase
    masks[:, 7] = ~pair_phase
    masks[:, :6] = valid
    return masks


def optimize_batch(online: QNetwork, target: QNetwork,
                   optimizer: torch.optim.Optimizer, replay: ReplayBuffer,
                   config: DQNConfig, device: torch.device,
                   reference: QNetwork | None = None,
                   reference_weight: float = 0.0,
                   reference_temperature: float = 1.0) -> torch.Tensor:
    """Perform one update, returning a detached loss without device sync."""
    prioritized = isinstance(replay, PrioritizedReplayBuffer)
    sampled = replay.sample(
        config.batch_size, device, include_n_steps=True,
        include_priorities=prioritized,
        include_counterfactual=config.counterfactual_loss_weight > 0.0,
    )
    if prioritized:
        if config.counterfactual_loss_weight > 0.0:
            (states, actions, rewards, next_states, dones, n_steps,
             counterfactual, counterfactual_mask, weights, indices) = sampled
        else:
            states, actions, rewards, next_states, dones, n_steps, weights, indices = sampled
    else:
        if config.counterfactual_loss_weight > 0.0:
            (states, actions, rewards, next_states, dones, n_steps,
             counterfactual, counterfactual_mask) = sampled
        else:
            states, actions, rewards, next_states, dones, n_steps = sampled
    # Reuse this forward pass for TD, reference-policy, and counterfactual
    # losses.  The previous implementation evaluated the same states up to
    # three times per optimizer update.
    q_values = online(states)
    predicted = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_values_all = online(next_states) if config.pairing_policy == "learned" else None
        next_target_all = target(next_states) if config.pairing_policy == "learned" else None
        if next_values_all is not None and next_target_all is not None:
            next_mask = joint_legal_action_mask(next_states)
            next_values_all = next_values_all.masked_fill(~next_mask, float("-inf"))
            next_target_all = next_target_all.masked_fill(~next_mask, float("-inf"))
        if config.algorithm == "double-dqn":
            next_actions = (next_values_all if next_values_all is not None
                            else online(next_states)).argmax(dim=1, keepdim=True)
            next_values = (next_target_all if next_target_all is not None
                           else target(next_states)).gather(1, next_actions).squeeze(1)
        else:
            next_values = (next_target_all if next_target_all is not None
                           else target(next_states)).max(dim=1).values
        discounts = torch.pow(
            torch.full_like(rewards, config.gamma), n_steps.float()
        )
        expected = rewards + discounts * (~dones).float() * next_values
    td_errors = expected - predicted
    per_sample_loss = nn.functional.smooth_l1_loss(
        predicted, expected, reduction="none"
    )
    if prioritized:
        td_loss = (weights * per_sample_loss).mean()
    else:
        td_loss = per_sample_loss.mean()
    loss = td_loss
    if reference is not None and reference_weight > 0.0:
        if reference_temperature <= 0.0:
            raise ValueError("reference_temperature must be positive")
        with torch.no_grad():
            reference_logits = reference(states) / reference_temperature
        policy_logits = q_values / reference_temperature
        policy_loss = nn.functional.kl_div(
            nn.functional.log_softmax(policy_logits, dim=1),
            nn.functional.softmax(reference_logits, dim=1),
            reduction="batchmean",
        ) * (reference_temperature ** 2)
        loss = loss + reference_weight * policy_loss
    if config.counterfactual_loss_weight > 0.0:
        # Counterfactual labels use the compact [Continue, Stop] layout even
        # when the joint learner also has six pairing actions.
        continue_index = 6 if config.pairing_policy == "learned" else 0
        stop_index = 7 if config.pairing_policy == "learned" else 1
        q_gap = q_values[:, continue_index] - q_values[:, stop_index]
        target_gap = (
            counterfactual[:, 0] - counterfactual[:, 1]
        ) * config.counterfactual_target_scale
        auxiliary = nn.functional.smooth_l1_loss(q_gap, target_gap, reduction="none")
        mask = counterfactual_mask.float()
        if prioritized:
            auxiliary = weights * auxiliary
        loss = loss + config.counterfactual_loss_weight * (
            (auxiliary * mask).sum() / mask.sum().clamp_min(1.0)
        )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), config.gradient_clip)
    optimizer.step()
    if prioritized:
        replay.update_priorities(indices, td_errors)
    return loss.detach()


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_checkpoint(path: str | Path, online: QNetwork, target: QNetwork,
                    optimizer: torch.optim.Optimizer, config: DQNConfig,
                    *, global_step: int, episodes: int,
                    metadata: dict[str, object] | None = None) -> None:
    """Save a device-portable training checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_version": CHECKPOINT_VERSION,
        "config": asdict(config),
        "online_state_dict": _cpu_state_dict(online),
        "target_state_dict": _cpu_state_dict(target),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "episodes": episodes,
        "metadata": metadata or {},
    }, destination)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, object]:
    """Load only tensor/primitive checkpoint data, remapped to the chosen device."""
    kwargs: dict[str, object] = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    checkpoint = torch.load(Path(path), **kwargs)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    return checkpoint


class DQNAgent:
    """Greedy neural policy, optionally selecting dice pairings as well."""

    def __init__(self, model: QNetwork, *, enhanced_features: bool,
                  device: torch.device, pair_heuristic: str = "ga",
                  domain_features: bool = False,
        strategic_features: bool = False) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.enhanced_features = enhanced_features
        self.domain_features = domain_features
        self.strategic_features = strategic_features
        self.learn_pairing = pair_heuristic == "learned"
        self.device = device
        if pair_heuristic == "ga":
            self.pairing_policy = Rule28Agent(pair_heuristic="ga", smart=True)
        elif pair_heuristic == "couto":
            from .agents import CoutoAgent
            self.pairing_policy = CoutoAgent(pair_heuristic="couto")
        elif pair_heuristic != "learned":
            raise ValueError("pair_heuristic must be 'ga', 'couto', or 'learned'")

    def _features(self, state: State, moves: Sequence[Move] = ()) -> list[float]:
        features = state.features(
            enhanced=self.enhanced_features, domain=self.domain_features,
            strategic=self.strategic_features,
        )
        if self.learn_pairing:
            values: list[float] = []
            for move in sorted(moves)[:6]:
                values.extend((move.columns[0] / 12.0,
                               move.columns[1] / 12.0 if len(move.columns) == 2 else 0.0,
                               1.0))
            values.extend([0.0] * (18 - len(values)))
            features += values
        return features

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "DQNAgent":
        selected = select_device(device)
        checkpoint = load_checkpoint(path, selected)
        values = dict(checkpoint["config"])
        values["hidden_sizes"] = tuple(values["hidden_sizes"])
        config = DQNConfig(**values)
        model = QNetwork(config.observation_size, config.action_size,
                          config.hidden_sizes, config.network_type).to(selected)
        model.load_state_dict(checkpoint["online_state_dict"])
        if config.observation_size not in (58, 60, 68, 80, 76, 78, 86, 98):
            raise ValueError(
                "Checkpoint observation size is not supported"
            )
        base_observation_size = (
            config.observation_size
            - (18 if config.pairing_policy == "learned" else 0)
        )
        return cls(
            model,
            enhanced_features=base_observation_size >= 60,
            domain_features=base_observation_size >= 68,
            strategic_features=base_observation_size == 80,
            pair_heuristic=config.pairing_policy,
            device=selected,
        )

    def choose_move(self, state: State, moves: Sequence[Move],
                    rng: random.Random) -> Move:
        if self.learn_pairing:
            ordered = sorted(moves)
            features = self._features(state, ordered)
            with torch.no_grad():
                values = self.model(torch.tensor(
                    features, dtype=torch.float32, device=self.device
                ).unsqueeze(0))[0]
            return ordered[int(values[:len(ordered)].argmax().item())]
        return self.pairing_policy.choose_move(state, moves, rng)

    def should_stop(self, state: State, rng: random.Random) -> bool:
        del rng
        features = self._features(state)
        with torch.no_grad():
            observation = torch.tensor(features, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
            values = self.model(observation)
            if self.learn_pairing:
                action = int(values[:, 6:].argmax(dim=1).item()) + 6
            else:
                action = int(values.argmax(dim=1).item())
        return action == int(JointAction.STOP if self.learn_pairing else StopContinueAction.STOP)
