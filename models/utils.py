import sys

from models.gaussian_policy import GaussianPolicy
from models.deterministic_policy import DeterministicPolicy


def init_policy(policy_type, model_args, wwid):
    if policy_type == 'Deterministic':
        policy = DeterministicPolicy(**model_args, wwid=wwid)
    elif policy_type == 'Gaussian':
        policy = GaussianPolicy(**model_args, wwid=wwid)
    else:
        print('Not implemented policy type: {}'.format(policy_type))
        sys.exit()

    return policy
