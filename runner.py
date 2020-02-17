import os
import time
import datetime

import torch
import numpy as np
import gym

from cerl import CERL
from logger import Logger


class Parameters:
    def __init__(self):
        self.env_name = 'HalfCheetah-v2'
        self.seed = 1

        # Learner's param
        self.algo = 'TD3'
        self.policy_type = 'Deterministic'
        self.capacity = 1000000
        self.batch_size = 256   # Batch size
        self.rollout_size = 10    # Size of learner rollouts
        dummy_env = gym.make(self.env_name)
        self.state_dim = dummy_env.observation_space.shape[0]
        self.action_dim = dummy_env.action_space.shape[0]
        self.hidden_sizes = [400, 300]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ucb_coefficient = 0.9  # Exploration coefficient in UCB

        # neuroevolution param
        self.pop_size = 10
        self.elite_fraction = 0.2
        self.crossover_prob = 0.01
        self.mutation_prob = 0.2
        self.weight_magnitude_limit = 10000000

        self.test_size = 10

        # Save Results
        self.savefolder = 'results/'
        os.makedirs('results/', exist_ok=True)


def main():
    args = Parameters()

    exp_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(parent_dir, 'results', exp_name)
    logger = Logger(save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = CERL(args)
    print('Running CERL for', args.env_name, 'State_dim:', args.state_dim, ' Action_dim:', args.action_dim)

    start_time = time.time()
    for gen in range(1, 10000000):

        pop_fitness, learner_fitness, allocation_count, test_mean, test_std, champ_wwid = agent.train(gen, logger)

        print('Gen {}, Frames {}, Frames/sec: {:.2f}'.format(gen, agent.total_frames, agent.total_frames / (time.time() - start_time)))
        print('Population Fitness', ['{:.3f}'.format(fitness) for fitness in pop_fitness])
        print('Learner Average Fitness', ['{:.3f}'.format(fitness) for fitness in learner_fitness])
        print('Resource Allocation', allocation_count)
        try:
            print('Best Policy ever genealogy:', agent.genealogy.tree[int(agent.best_policy.wwid.item())].history)
            print('Champ genealogy:', agent.genealogy.tree[champ_wwid].history)
        except KeyError:
            pass
        print()

        if test_mean is not None:
            print('====================')
            print('Test_score for the champion: {:.3f} ({:.3f})'.format(test_mean, test_std))
            print('====================')

        if agent.total_frames > 10000:
            break
    logger.save()

    # Kill all processes
    try:
        for p in agent.task_pipes:
            p[0].send('TERMINATE')
        for p in agent.test_task_pipes:
            p[0].send('TERMINATE')
        for p in agent.evo_task_pipes:
            p[0].send('TERMINATE')
    except:
        pass


if __name__ == '__main__':
    main()