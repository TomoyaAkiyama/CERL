import sys

import torch

from learner import Learner


td3_kwargs = {
        'tau': 5e-3,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'exploration_noise': 0.1,
        'policy_freq': 2
    }


def init_portfolio(state_dim, action_dim, use_cuda, genealogy, portfolio_name):
    if portfolio_name == 'portfolio1':
        lr = 1e-3
        gammas = [0.9, 0.99, 0.997, 0.9995]
        policy_type = 'Deterministic'
        model_args = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_sizes': [400, 300],
            'activation': 'ReLU',
            'layernorm': False
        }

        if use_cuda:
            devices = []
            gpu_num = torch.cuda.device_count()
            for i in range(len(gammas)):
                devices.append(torch.device('cuda:{}'.format(i % gpu_num)))
        else:
            devices = [torch.device('cpu')] * len(gammas)
        portfolio = []
        for i, (device, gamma) in enumerate(zip(devices, gammas)):
            wwid = genealogy.new_id('Learner_{}'.format(i))
            portfolio.append(
                Learner('TD3', model_args, wwid, device, lr, gamma, **td3_kwargs)
            )
    else:
        print('There is no portfolio named {}'.format(portfolio_name))
        sys.exit()

    return portfolio, policy_type, model_args
