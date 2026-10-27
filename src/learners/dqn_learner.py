"""Environment-independent DQN optimization owner."""
from __future__ import annotations

import torch

from cant_stop.dqn import (
    DQNConfig,
    PrioritizedReplayBuffer,
    QNetwork,
    ReplayBuffer,
    optimize_batch,
)


class DQNLearner:
    """Own optimization decisions without knowing games or opponents."""

    def __init__(
        self,
        online: QNetwork,
        target: QNetwork,
        optimizer: torch.optim.Optimizer,
        replay: ReplayBuffer,
        config: DQNConfig,
        device: torch.device,
        *,
        reference: QNetwork | None = None,
        reference_weight: float = 0.0,
        reference_temperature: float = 1.0,
    ) -> None:
        self.online = online
        self.target = target
        self.optimizer = optimizer
        self.replay = replay
        self.config = config
        self.device = device
        self.reference = reference
        self.reference_weight = reference_weight
        self.reference_temperature = reference_temperature

    def optimize_due(
        self, global_step: int, gradient_updates: int
    ) -> tuple[int, list[torch.Tensor]]:
        """Perform every update due at ``global_step`` and return detached losses."""
        if (global_step < self.config.warmup_steps
                or len(self.replay) < self.config.batch_size):
            return gradient_updates, []
        if isinstance(self.replay, PrioritizedReplayBuffer):
            fraction = min(
                global_step / max(self.config.epsilon_decay_steps, 1), 1.0
            )
            self.replay.set_beta(
                self.config.per_beta_start + fraction * (
                    self.config.per_beta_end - self.config.per_beta_start
                )
            )
        desired_updates = 1 + (
            global_step - self.config.warmup_steps
        ) // self.config.train_frequency
        losses = []
        while gradient_updates < desired_updates:
            losses.append(optimize_batch(
                self.online, self.target, self.optimizer, self.replay,
                self.config, self.device, self.reference,
                self.reference_weight, self.reference_temperature,
            ))
            gradient_updates += 1
        return gradient_updates, losses

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
