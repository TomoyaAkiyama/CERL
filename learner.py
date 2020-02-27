from off_policy_algorithms.utils import init_algo


class Learner:
    def __init__(
            self,
            algo_name,
            model_args,
            wwid,
            device,
            lr,
            gamma,
            **kwargs
    ):

        self.algo = init_algo(algo_name, model_args, wwid, device, lr, gamma, **kwargs)
        self.fitnesses = []
        self.ep_lens = []
        self.value = None
        self.visit_count = 0

    def update_parameters(self, replay_buffer, batch_size, iteration):
        for _ in range(iteration):
            self.algo.train(replay_buffer, batch_size)

    def update_stats(self, fitness, ep_len, kappa=0.2):
        self.visit_count += 1
        self.fitnesses.append(fitness)
        self.ep_lens.append(ep_len)

        if self.value is None:
            self.value = fitness
        else:
            self.value = kappa * fitness + (1. - kappa) * self.value
