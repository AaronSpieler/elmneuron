import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import jit

from .modeling_utils import (
    MLP,
    create_interlocking_indices,
    create_overlapping_window_indices,
    custom_tanh,
    inverse_scaled_sigmoid,
    scaled_sigmoid,
)
from .neuronio.neuronio_data_utils import DEFAULT_Y_TRAIN_SOMA_SCALE

PREPROCESS_CONFIGURATIONS = [None, "random_routing", "neuronio_routing"]


class ELM(jit.ScriptModule):
    """Expressive Leaky Memory (ELM) neuron model."""

    __constants__ = [
        "num_input",
        "num_output",
        "num_memory",
        "lambda_value",
        "mlp_num_layers",
        "mlp_activation",
        "memory_tau_min",
        "memory_tau_max",
        "learn_memory_tau",
        "w_s_value",
        "num_synapse_per_branch",
        "input_to_synapse_routing",
        "delta_t",
    ]

    def __init__(
        self,
        num_input: int,
        num_output: int,
        num_memory: int = 100,
        lambda_value: float = 5.0,
        mlp_num_layers: int = 1,
        mlp_hidden_size: Optional[int] = None,
        mlp_activation: str = "relu",
        tau_s_value: float = 5.0,
        memory_tau_min: float = 1.0,
        memory_tau_max: float = 1000.0,
        learn_memory_tau: bool = False,
        w_s_value: float = 0.5,
        num_branch: Optional[int] = None,
        num_synapse_per_branch: int = 1,
        input_to_synapse_routing: Optional[str] = None,
        delta_t: float = 1.0,
    ):
        super(ELM, self).__init__()
        self.num_input, self.num_output = num_input, num_output
        self.num_memory = num_memory
        self.lambda_value = lambda_value
        self.mlp_num_layers = mlp_num_layers
        self.mlp_activation = mlp_activation
        self.memory_tau_min, self.memory_tau_max = memory_tau_min, memory_tau_max
        self.learn_memory_tau = learn_memory_tau
        self.tau_s_value, self.w_s_value = tau_s_value, w_s_value
        self.num_synapse_per_branch = num_synapse_per_branch
        self.input_to_synapse_routing = input_to_synapse_routing
        self.delta_t = delta_t

        # derived neuron properties
        self.mlp_hidden_size = mlp_hidden_size if mlp_hidden_size else 2 * num_memory
        self.num_branch = self.num_input if num_branch is None else num_branch
        self.num_mlp_input = self.num_branch + num_memory
        self.num_synapse = num_synapse_per_branch * self.num_branch

        # sanity check of input configuration
        assert self.num_synapse == num_input or input_to_synapse_routing is not None
        assert self.input_to_synapse_routing in PREPROCESS_CONFIGURATIONS

        # initialization of model weights
        self.mlp = MLP(
            self.num_mlp_input,
            self.mlp_hidden_size,
            num_memory,
            mlp_num_layers,
            mlp_activation,
        )
        self._proto_w_s = nn.parameter.Parameter(
            torch.full((self.num_synapse,), w_s_value)
        )
        self.w_y = nn.Linear(num_memory, num_output)

        # initialization of synapse time constants and decay factors
        tau_s = torch.full((self.num_synapse,), tau_s_value)
        self.tau_s = nn.parameter.Parameter(tau_s, requires_grad=False)

        # initialization of memory time constants and decay factors
        _proto_tau_m = torch.logspace(
            math.log10(memory_tau_min + 1e-6),
            math.log10(memory_tau_max - 1e-6),
            num_memory,
        )
        _proto_tau_m = inverse_scaled_sigmoid(
            _proto_tau_m, memory_tau_min, memory_tau_max
        )
        self._proto_tau_m = nn.parameter.Parameter(
            _proto_tau_m, requires_grad=learn_memory_tau
        )

        # NOTE: part of model for ease of use
        routing_artifacts = self.create_input_to_synapse_indices()
        self.input_to_synapse_indices = nn.parameter.Parameter(
            routing_artifacts[0], requires_grad=False
        )
        self.valid_indices_mask = nn.parameter.Parameter(
            routing_artifacts[1], requires_grad=False
        )

    @property
    def tau_m(self):
        return scaled_sigmoid(
            self._proto_tau_m, self.memory_tau_min, self.memory_tau_max
        )

    @property
    def kappa_m(self):
        return torch.exp(-self.delta_t / torch.clamp(self.tau_m, min=1e-6))

    @property
    def kappa_s(self):
        return torch.exp(-self.delta_t / torch.clamp(self.tau_s, min=1e-6))

    @property
    def w_s(self):
        return torch.relu(self._proto_w_s)

    # NOTE: part of model for ease of use
    def create_input_to_synapse_indices(self):
        if self.input_to_synapse_routing == "random_routing":
            # randomly select num_synapse from num_input
            input_to_synapse_indices = torch.randint(
                self.num_input, (self.num_synapse,)
            )
            return input_to_synapse_indices, torch.ones_like(input_to_synapse_indices)
        elif self.input_to_synapse_routing == "neuronio_routing":
            # sanity check of input configuration
            assert (
                math.ceil(self.num_input / self.num_branch)
                <= self.num_synapse_per_branch
            )

            # interlace excitatory and inhibitory inputs
            interlocking_indices = create_interlocking_indices(self.num_input)
            # assign neighbouring inputs to same branch
            overlapping_indices, valid_indices_mask = create_overlapping_window_indices(
                self.num_input, self.num_branch, self.num_synapse_per_branch
            )
            input_to_synapse_indices = interlocking_indices[overlapping_indices]

            return input_to_synapse_indices, valid_indices_mask
        else:
            return None, None

    # NOTE: part of model for ease of use
    def route_input_to_synapses(self, x):
        if self.input_to_synapse_routing is not None:
            x = torch.index_select(x, 2, self.input_to_synapse_indices)
            x = x * self.valid_indices_mask  # valid mask
        return x

    @jit.script_method
    def dynamics(self, x, s_prev, m_prev, w_s, kappa_s, kappa_m):
        # compute the dynamics for a single timestep
        batch_size, _ = x.shape
        s_t = kappa_s * s_prev + w_s * x
        syn_input = s_t.view(batch_size, self.num_branch, -1).sum(dim=-1)
        delta_m_t = custom_tanh(
            self.mlp(torch.cat([syn_input, kappa_m * m_prev], dim=-1))
        )
        m_t = kappa_m * m_prev + self.lambda_value * (1 - kappa_m) * delta_m_t
        y_t = self.w_y(m_t)
        return y_t, s_t, m_t

    @jit.script_method
    def forward(self, X):
        # compute the the recurrent dynamics for a sample
        batch_size, T, _ = X.shape
        w_s = self.w_s
        kappa_s, kappa_m = self.kappa_s, self.kappa_m
        s_prev = torch.zeros(batch_size, len(kappa_s), device=X.device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=X.device)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        inputs = self.route_input_to_synapses(X)
        for t in range(T):
            y_t, s_prev, m_prev = self.dynamics(
                inputs[:, t], s_prev, m_prev, w_s, kappa_s, kappa_m
            )
            outputs.append(y_t)
        return torch.stack(outputs, dim=-2)

    # NOTE: part of model for ease of use
    @jit.script_method
    def neuronio_eval_forward(
        self, X, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ):
        outputs = self.forward(X)
        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]

        # apply sigmoid to spike (probability) prediction
        spike_pred = torch.sigmoid(spike_pred)
        # apply soma scale to soma prediction
        soma_pred = 1 / y_train_soma_scale * soma_pred

        return torch.stack([spike_pred, soma_pred], dim=-1)

    # NOTE: part of model for ease of use
    @jit.script_method
    def neuronio_viz_forward(
        self, X, y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE
    ):
        # compute the the recurrent dynamics for a sample
        batch_size, T, _ = X.shape
        w_s = self.w_s
        kappa_s, kappa_m = self.kappa_s, self.kappa_m
        s_prev = torch.zeros(batch_size, len(kappa_s), device=X.device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=X.device)

        # calcualte the outputs, synapse and memory values
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        s_record = torch.jit.annotate(List[torch.Tensor], [])
        m_record = torch.jit.annotate(List[torch.Tensor], [])
        inputs = self.route_input_to_synapses(X)
        for t in range(T):
            y_t, s_prev, m_prev = self.dynamics(
                inputs[:, t], s_prev, m_prev, w_s, kappa_s, kappa_m
            )
            outputs.append(y_t)
            s_record.append(s_prev)
            m_record.append(m_prev)
        outputs = torch.stack(outputs, dim=-2)
        s_record = torch.stack(s_record, dim=-2)
        m_record = torch.stack(m_record, dim=-2)

        # postprocess the outputs
        spike_pred, soma_pred = outputs[..., 0], outputs[..., 1]
        spike_pred = torch.sigmoid(spike_pred)
        soma_pred = 1 / y_train_soma_scale * soma_pred
        outputs = torch.stack([spike_pred, soma_pred], dim=-1)

        return outputs, s_record, m_record
