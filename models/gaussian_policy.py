import torch
import torch.nn as nn
import torch.distributions

from models.model_utils import create_hidden_layers

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class GaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_sizes,
            activation='ReLU',
            layernorm=False,
            wwid=-1,
    ):
        super(GaussianPolicy, self).__init__()

        self.wwid = torch.tensor([wwid])

        shared_layers = create_hidden_layers(state_dim, hidden_sizes, activation, layernorm)
        self.shared_layers = nn.Sequential(*shared_layers)
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, state):
        state = self.shared_layers(state)
        loc = self.mean_layer(state)
        log_std = self.log_std_layer(state)
        log_std = log_std.clamp(min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return loc, log_std

    def sample(self, state):
        loc, log_std = self.forward(state)
        scale = log_std.exp()
        dist = torch.distributions.Normal(loc=loc, scale=scale)
        ac = dist.rsample()
        action = torch.tanh(ac)
        action_log_prob = dist.log_prob(ac)
        action_log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        action_log_prob = action_log_prob.sum(-1, keepdim=True)

        return action, action_log_prob

    def deterministic_action(self, state):
        action, _ = self.forward(state)
        return action.cpu().detach().numpy().flatten()
