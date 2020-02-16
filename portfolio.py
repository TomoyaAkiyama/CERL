from learner import Learner


def init_portfolio(state_dim, action_dim, device, genealogy):
    algo_kwargs = {
        'tau': 5e-3,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'exploration_noise': 0.1,
        'policy_freq': 2
    }
    hidden_sizes = [400, 300]
    lr = 1e-3

    portfolio = []

    wwid = genealogy.new_id('learner_1')
    portfolio.append(
        Learner('TD3', state_dim, action_dim, hidden_sizes, device, lr, gamma=0.9, wwid=wwid, **algo_kwargs)
    )
    wwid = genealogy.new_id('learner_2')
    portfolio.append(
        Learner('TD3', state_dim, action_dim, hidden_sizes, device, lr, gamma=0.99, wwid=wwid, **algo_kwargs)
    )
    wwid = genealogy.new_id('learner_3')
    portfolio.append(
        Learner('TD3', state_dim, action_dim, hidden_sizes, device, lr, gamma=0.997, wwid=wwid, **algo_kwargs)
    )
    wwid = genealogy.new_id('learner_4')
    portfolio.append(
        Learner('TD3', state_dim, action_dim, hidden_sizes, device, lr, gamma=0.9995, wwid=wwid, **algo_kwargs)
    )

    return portfolio
