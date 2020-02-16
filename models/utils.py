import sys

from models.gaussian_policy import GaussianPolicy
from models.deterministic_policy import DeterministicPolicy


def init_policy(state_dim, action_dim, hidden_sizes, wwid, policy_type):
    if policy_type == 'Deterministic':
        policy = DeterministicPolicy(state_dim, action_dim, hidden_sizes, wwid)
    elif policy_type == 'Gaussian':
        policy = GaussianPolicy(state_dim, action_dim, hidden_sizes, wwid)
    else:
        print('Not implemented policy type: {}'.format(policy_type))
        sys.exit()

    return policy
