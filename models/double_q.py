import torch
import torch.nn as nn

from models.model_utils import create_hidden_layers


class DoubleQ(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_sizes,
            activation='ReLU',
            layernorm=False
    ):
        super(DoubleQ, self).__init__()

        input_dim = state_dim + action_dim

        layers = create_hidden_layers(input_dim, hidden_sizes, activation, layernorm)
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.Q1 = nn.Sequential(*layers)

        layers = create_hidden_layers(input_dim, hidden_sizes, activation, layernorm)
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.Q2 = nn.Sequential(*layers)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        q1 = self.Q1.forward(state_action)
        q2 = self.Q2.forward(state_action)

        return q1, q2

    def q1(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        q1 = self.Q1.forward(state_action)

        return q1
