import sys

from off_policy_algorithms import TD3, SAC


def init_algo(
        algo_name,
        model_args,
        wwid,
        device,
        lr,
        gamma,
        **kwargs,
):

    algo_kwargs = {
        'model_args': model_args,
        'wwid': wwid,
        'device': device,
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
