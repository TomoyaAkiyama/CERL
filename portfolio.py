import torch

from learner import Learner


def init_portfolio(state_dim, action_dim, use_cuda, genealogy):
    algo_kwargs = {
        'tau': 5e-3,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'exploration_noise': 0.1,
        'policy_freq': 2
    }
    hidden_sizes = [400, 300]
    lr = 1e-3

    if use_cuda:
        devices = []
        gpu_num = torch.cuda.device_count()
        for i in range(4):
            devices.append(torch.device('cuda:{}'.format(i % gpu_num)))
    else:
        devices = [torch.device('cpu')] * 4

    portfolio = []

    wwid = genealogy.new_id('Learner_1')
    portfolio.append(
        Learner('TD3', state_dim, action_dim, hidden_sizes, devices[0], lr, gamma=0.9, wwid=wwid, **algo_kwargs)
    )
    wwid = genealogy.new_id('Learner_2')
    portfolio.append(
        Learner('TD3', state_dim, action_dim, hidden_sizes, devices[1], lr, gamma=0.99, wwid=wwid, **algo_kwargs)
    )
    wwid = genealogy.new_id('Learner_3')
    portfolio.append(
        Learner('TD3', state_dim, action_dim, hidden_sizes, devices[2], lr, gamma=0.997, wwid=wwid, **algo_kwargs)
    )
    wwid = genealogy.new_id('Learner_4')
    portfolio.append(
        Learner('TD3', state_dim, action_dim, hidden_sizes, devices[3], lr, gamma=0.9995, wwid=wwid, **algo_kwargs)
    )

    return portfolio
