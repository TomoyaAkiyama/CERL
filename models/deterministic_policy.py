import torch
import torch.nn as nn

from models.model_utils import create_hidden_layers


class DeterministicPolicy(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_sizes,
            wwid,
            activation='ReLU'
    ):
        super(DeterministicPolicy, self).__init__()

        self.wwid = torch.tensor([wwid])

        layers = create_hidden_layers(state_dim, hidden_sizes, activation)
        layers.append(nn.Linear(hidden_sizes[-1], action_dim))
        layers.append(nn.Tanh())
        self.actor = nn.Sequential(*layers)

    def forward(self, state):
        action = self.actor.forward(state)
        return action

    def deterministic_action(self, state):
        action = self.forward(state)
        return action.cpu().detach().numpy().flatten()
