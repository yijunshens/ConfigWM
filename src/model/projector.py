"""Configurable projector heads for Configurable Encoder encoders."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_projector(
    input_dim: int,
    config: dict[str, Any],
) -> nn.Module:
    """Build a projector from config.

    Config keys:
        type: "linear" or "mlp"
        dim: output dimension
        hidden_dim: hidden dimension (mlp only, default 2048)
        num_layers: number of linear layers (mlp only, default 2)
            - 2: input → hidden → output (LeJEPA default)
            - 3: input → hidden → hidden → output
            - Each hidden layer has BN + ReLU. Last layer is raw (no BN/ReLU).
    """
    proj_type = config.get("type", "mlp")
    output_dim = int(config["dim"])
    hidden_dim = int(config.get("hidden_dim", 2048))
    num_layers = int(config.get("num_layers", 2))

    if proj_type == "linear":
        proj = nn.Linear(input_dim, output_dim)
        nn.init.normal_(proj.weight, std=1e-5)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)
        return proj

    if proj_type == "mlp":
        use_bn = config.get("batch_norm", True)
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        # Last layer: raw linear (no BN, no ReLU — matching LeJEPA)
        last_bias = config.get("last_layer_bias", True)
        layers.append(nn.Linear(in_dim, output_dim, bias=last_bias))
        return nn.Sequential(*layers)

    raise ValueError(f"Unsupported projector type: {proj_type}")
