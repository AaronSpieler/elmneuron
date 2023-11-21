import math

import torch
import torch.nn as nn


def scaled_sigmoid(x, lower_bound: float, upper_bound: float):
    return (upper_bound - lower_bound) * torch.sigmoid(x) + lower_bound


def inverse_scaled_sigmoid(x, lower_bound: float, upper_bound: float):
    x = torch.clamp(x, lower_bound + 1e-6, upper_bound - 1e-6)
    return torch.log((x - lower_bound) / (upper_bound - x))


def custom_tanh(x):
    return torch.tanh(x * 2 / 3) * 1.7159


def create_interlocking_indices(num_input: int):
    half_num_input_data = num_input // 2
    half_range_steps = (torch.arange(num_input) % 2) * half_num_input_data
    single_steps = torch.div(torch.arange(num_input), 2, rounding_mode="floor")
    return half_range_steps + single_steps


def create_overlapping_window_indices(
    num_input: int, num_windows: int, num_elements_per_window: int
):
    stride_size = math.ceil(num_input / num_windows)
    overlapping_indices = (
        torch.arange(num_windows).unsqueeze(1) * stride_size
    ) + torch.arange(num_elements_per_window).unsqueeze(0)
    valid_indices = overlapping_indices < num_input
    overlapping_indices = torch.clamp(overlapping_indices, max=num_input - 1)  # fix
    return overlapping_indices.flatten(), valid_indices.flatten()


class MLP(nn.Module):
    """Multi-layer perceptron (MLP) model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_output: int,
        num_layers: int,
        activation: str = "relu",
    ):
        super(MLP, self).__init__()
        next_input_size = input_size

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(next_input_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            next_input_size = hidden_size

        layers.append(nn.Linear(next_input_size, num_output))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
