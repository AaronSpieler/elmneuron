import math
import torch
import torch.nn as nn

# NOTE: the 1e-6 used throughout are to avoid numerical instabilities
# NOTE: the tau_m bounds are enforced using a scaled sigmoid
# NOTE: the w_s bounds are enforced using relu (originally exp)

def scaled_sigmoid(x, lower_bound, upper_bound):
    return (upper_bound - lower_bound) * torch.sigmoid(x) + lower_bound

def inverse_scaled_sigmoid(x, lower_bound, upper_bound):
    x = torch.clamp(x, lower_bound + 1e-6, upper_bound - 1e-6)
    return torch.log((x - lower_bound) / (upper_bound - x))

def custom_tanh(x):
    return torch.tanh(x * 2 / 3) * 1.7159

class MLP(nn.Module):
    """Multi-layer perceptron (MLP) model."""

    def __init__(self, input_size, hidden_size, num_output, num_layers):
        super(MLP, self).__init__()
        next_input_size = input_size

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(next_input_size, hidden_size))
            layers.append(nn.ReLU())
            next_input_size = hidden_size

        layers.append(nn.Linear(next_input_size, num_output))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ELM(nn.Module):
    """Expressive Leaky Memory (ELM) neuron model."""
    
    def __init__(self, num_synapse, num_output, num_memory=100, 
                 mlp_num_layers=1, memory_tau_min=10, memory_tau_max=1000, 
                 lambda_value=3.0, tau_s_value=5.0, w_s_value=0.5, delta_t=1.0, 
                 learn_memory_tau=False,mlp_hidden_size=None,use_num_branch=None):
        super(ELM, self).__init__()
        self.delta_t = delta_t
        self.lambda_value = lambda_value
        self.learn_memory_tau = learn_memory_tau
        self.memory_tau_min = memory_tau_min
        self.memory_tau_max = memory_tau_max

        # configuration in case of using branches
        self.use_num_branch = use_num_branch
        if use_num_branch is not None:
            assert num_synapse % use_num_branch == 0, "must be exactly divisible"
            num_mlp_input = use_num_branch + num_memory
        else:
            num_mlp_input = num_synapse + num_memory

        # initialization of model parameters
        mlp_hidden_size = mlp_hidden_size if mlp_hidden_size else 2 * num_memory
        self.mlp = MLP(num_mlp_input, mlp_hidden_size, num_memory, mlp_num_layers)
        self.w_s = nn.parameter.Parameter(torch.full((num_synapse,), w_s_value))
        self.w_y = nn.Linear(num_memory, num_output)

        # initialization of time constants and decay factors
        self.tau_s = torch.full((num_synapse,), tau_s_value)
        self.kappa_s = torch.exp(-delta_t / torch.clamp(self.tau_s, min=1e-6))
        self._proto_tau_m = torch.logspace(
            math.log10(memory_tau_min + 1e-6),
            math.log10(memory_tau_max - 1e-6),
            num_memory,
        )
        self._proto_kappa_m = torch.exp(
            -delta_t / torch.clamp(self._proto_tau_m, min=1e-6)
        )

        # configuration in case memory tau is learnable
        if self.learn_memory_tau:
            self._proto_tau_m = inverse_scaled_sigmoid(
                self._proto_tau_m, memory_tau_min, memory_tau_max
            )
            self._proto_tau_m = nn.parameter.Parameter(self._proto_tau_m)

    @property
    def tau_m(self):
        # ensure that tau_m is always up to date
        if self.learn_memory_tau:
            return scaled_sigmoid(
                self._proto_tau_m, self.memory_tau_min, self.memory_tau_max
            )
        else:
            return self._proto_tau_m

    @property
    def kappa_m(self):
        # ensure that kappa_m is always up to date
        if self.learn_memory_tau:
            return torch.exp(-self.delta_t / torch.clamp(self.tau_m, min=1e-6))
        else:
            return self._proto_kappa_m

    def dynamics(self, x, s_prev, m_prev, kappa_m):
        # compute the dynamics for a single timestep
        batch_size = x.shape[0]
        s_t = self.kappa_s * s_prev + torch.relu(self.w_s) * x
        if self.use_num_branch is not None:
            syn_input = s_t.view(batch_size, self.use_num_branch, -1).sum(dim=-1)
        else:
            syn_input = s_t
        delta_m_t = custom_tanh(
            self.mlp(torch.cat([syn_input, kappa_m * m_prev], dim=-1))
        )
        m_t = kappa_m * m_prev + self.lambda_value * (1 - kappa_m) * delta_m_t
        y_t = self.w_y(m_t)
        return y_t, s_t, m_t

    def forward(self, X):
        # compute the the recurrent dynamics for a sample
        batch_size, T, num_synapse = X.shape
        kappa_m = self.kappa_m
        s_prev = torch.zeros(batch_size, len(self.tau_s), device=X.device)
        m_prev = torch.zeros(batch_size, len(self.tau_m), device=X.device)
        outputs = []
        for t in range(T):
            y_t, s_prev, m_prev = self.dynamics(X[:, t], s_prev, m_prev, kappa_m)
            outputs.append(y_t)
        return torch.stack(outputs, dim=-2)
