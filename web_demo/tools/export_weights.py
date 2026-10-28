"""Export the online DQN MLP from a project checkpoint to browser JSON.

Run this from an environment with PyTorch installed. It intentionally exports
only inference weights, never the target network or optimizer state.
"""
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_weights.py CHECKPOINT OUTPUT_JSON")
    payload = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
    config = payload["config"]
    is_ppo = "model_state_dict" in payload
    if not is_ppo and config.get("network_type", "standard") != "standard":
        raise SystemExit("This exporter currently supports standard DQN only")
    state = payload["model_state_dict"] if is_ppo else payload["online_state_dict"]
    layers = []
    index = 0
    prefix = "trunk." if is_ppo else "network."
    while f"{prefix}{index}.weight" in state:
        weight = state[f"{prefix}{index}.weight"].tolist()
        bias = state[f"{prefix}{index}.bias"].tolist()
        layers.append({"weights": weight, "bias": bias})
        index += 2  # Linear, ReLU, Linear, ReLU, ...
    if not layers:
        raise SystemExit("No standard network layers found")
    if is_ppo:
        # Browser inference needs the policy head, not the scalar value head.
        # Append it as the final linear layer so the existing dependency-free
        # MLP loader can run PPO exactly like the Python actor.
        head = state.get("policy_head.weight")
        bias = state.get("policy_head.bias")
        if head is None or bias is None:
            raise SystemExit("PPO checkpoint has no policy_head")
        layers.append({"weights": head.tolist(), "bias": bias.tolist()})
    base_observation_size = config["observation_size"] - (
        18 if config.get("action_size", 2) == 8 else 0
    )
    output = {
        "format": "cant-stop-policy-v1",
        "observation_size": config["observation_size"],
        "action_size": config["action_size"],
        "features": ("strategic" if base_observation_size == 80
                     else "domain" if base_observation_size == 68
                     else "raw"),
        "activation": "tanh" if is_ppo else "relu",
        "pairing_policy": config.get("pairing_policy", "ga"),
        "layers": layers,
    }
    destination = Path(sys.argv[2]); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(layers)} layers to {destination}")


if __name__ == "__main__":
    main()
