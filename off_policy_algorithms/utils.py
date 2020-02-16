import sys

from off_policy_algorithms import TD3, SAC


def init_algo(
        algo_name,
        state_dim,
        action_dim,
        hidden_sizes,
        device,
        lr,
        gamma,
        wwid,
        **kwargs,
):

    algo_kwargs = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_sizes': hidden_sizes,
        'device': device,
        'wwid': wwid,
        'lr': lr,
        'gamma': gamma
    }

    if algo_name == 'TD3':
        algo_kwargs['tau'] = kwargs['tau']
        algo_kwargs['policy_noise'] = kwargs['policy_noise']
        algo_kwargs['noise_clip'] = kwargs['noise_clip']
        algo_kwargs['exploration_noise'] = kwargs['exploration_noise']
        algo_kwargs['policy_freq'] = kwargs['policy_freq']
        algo = TD3(**algo_kwargs)
    elif algo_name == 'SAC':
        algo_kwargs['tau'] = kwargs['tau']
        algo = SAC(**algo_kwargs)
    else:
        print('Off Policy Algorithm \"{}\" is not implemented.'.format(algo_name))
        sys.exit()

    return algo
