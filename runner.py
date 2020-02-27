import os
import time

import torch
import torch.backends.cudnn
import numpy as np

from cerl import CERL
from logger import Logger

np.set_printoptions(precision=3)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def main(env_name, seed, args):
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(
        parent_dir,
        'results',
        args['portfolio_id'],
        env_name,
        'seed{}'.format(seed)
    )
    logger = Logger(save_dir)

    set_seed(seed)

    args['env_name'] = env_name
    agent = CERL(**args)
    print('Running CERL for', env_name)

    try:
        start_time = time.time()
        gen = 0
        while True:
            gen += 1
            pop_fitness, learner_fitness, allocation_count, test_mean, test_std, champ_wwid = agent.train(gen, logger)

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
    args = {
        'rollout_size': 10,
        'pop_size': 10,
        'portfolio_id': 'portfolio1',
        'use_cuda': True,
        'capacity': 1000000,
        'batch_size': 256,
        'kappa': 0.2,
        'ucb_coefficient': 0.9,
        'elite_fraction': 0.2,
        'cross_prob': 0.01,
        'cross_fraction': 0.3,
        'bias_cross_prob': 0.2,
        'mutation_prob': 0.2,
        'mut_strength': 0.02,
        'mut_fraction': 0.03,
        'super_mut_prob': 0.1,
        'reset_prob': 0.2,
    }

    env_name = 'Swimmer-v2'
    seed = 1
    main(env_name, seed, args)
