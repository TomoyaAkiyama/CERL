import torch.nn as nn

activations = nn.ModuleDict({
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
})


def create_hidden_layers(input_dim, hidden_sizes, activation):
    layers = [
        nn.Linear(input_dim, hidden_sizes[0]),
        activations[activation],
    ]
    for i, hidden_size in enumerate(hidden_sizes[1:]):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(activations[activation])

    return layers


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
