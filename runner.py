import os
import time

import torch
import torch.backends.cudnn
import numpy as np
import gym

from cerl import CERL
from logger import Logger

np.set_printoptions(precision=3)


class Parameters:
    def __init__(self):
        self.env_name = 'HalfCheetah-v2'
        self.portfolio_id = 'portfolio1'
        self.seed = 1

        # Learner's param
        self.policy_type = 'Deterministic'
        self.capacity = 1000000
        self.batch_size = 256   # Batch size
        self.rollout_size = 10    # Size of learner rollouts
        self.ucb_coefficient = 0.9  # Exploration coefficient in UCB
        dummy_env = gym.make(self.env_name)
        self.state_dim = dummy_env.observation_space.shape[0]
        self.action_dim = dummy_env.action_space.shape[0]
        self.hidden_sizes = [400, 300]
        self.use_cuda = True

        # neuroevolution param
        self.pop_size = 10
        self.elite_fraction = 0.2
        self.crossover_prob = 0.01
        self.mutation_prob = 0.2
        self.weight_magnitude_limit = 10000000


def main(args):
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(
        parent_dir,
        'results',
        args.portfolio_id,
        args.env_name,
        'seed{}'.format(args.seed)
    )
    logger = Logger(save_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    agent = CERL(args)
    print('Running CERL for', args.env_name, 'State_dim:', args.state_dim, 'Action_dim:', args.action_dim)

    try:
        start_time = time.time()
        gen = 0
        while True:
            gen += 1
            pop_fitness, learner_fitness, allocation_count, test_mean, test_std, champ_wwid = agent.train(gen, logger)

            if test_mean is not None:
                message = 'Test Score of the Champ: {:.3f} ({:.3f})'.format(test_mean, test_std)
                print('=' * len(message))
                print(message)
                print('=' * len(message))
                print()

            print('Gen {}, Frames {},'.format(gen, agent.total_frames),
                  'Frames/sec: {:.2f}'.format(agent.total_frames / (time.time() - start_time)))
            print('\tPopulation Fitness', pop_fitness)
            print('\tLearner Average Fitness', learner_fitness)
            print('\tResource Allocation', allocation_count)
            print()
            print('\tBest Fitness ever: {:.3f}'.format(agent.best_score))
            print('\tBest Policy ever genealogy:', agent.genealogy.tree[int(agent.best_policy.wwid.item())].history)
            print('\tChamp Fitness: {:.3f}'.format(pop_fitness.max()))
            print('\tChamp genealogy:', agent.genealogy.tree[champ_wwid].history)
            print()

            if agent.total_frames >= 1000000:
                break
        logger.save()
        for i, learner in enumerate(agent.portfolio):
            model_dir = os.path.join(save_dir, 'learned_model', 'Learner{}'.format(i))
            os.makedirs(model_dir, exist_ok=True)
            learner.algo.save(model_dir)
        for i, individual in enumerate(agent.population):
            model_dir = os.path.join(save_dir, 'learned_model', 'Population')
            os.makedirs(model_dir, exist_ok=True)
            torch.save(individual.state_dict(), os.path.join(model_dir, 'individual{}.pth'.format(i)))

        print('Elapsed time: {}'.format(time.time() - start_time))
    finally:
        # Kill all processes
        for worker in agent.learner_workers:
            worker.terminate()
        for worker in agent.test_workers:
            worker.terminate()
        for worker in agent.evo_workers:
            worker.terminate()


if __name__ == '__main__':
    args = Parameters()
    for seed in range(1, 3):
        args.seed = seed
        main(args)
