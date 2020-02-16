import os
import time

import torch
import numpy as np
import gym

from tracker import Tracker
from cerl import CERL


class Parameters:
    def __init__(self):
        self.seed = 1
        self.env_name = 'HalfCheetah-v2'
        self.asynch_frac = 1.0  # Aynchronosity of NeuroEvolution
        self.algo = 'TD3'
        self.policy_type = 'Deterministic'
        self.capacity = 1000000

        self.test_size = 10

        self.batch_size = 256 # Batch size
        self.ucb_coefficient = 0.9  # Exploration coefficient in UCB
        self.rollout_size = 10    # Size of learner rollouts

        # NeuroEvolution stuff
        self.pop_size = 10
        self.elite_fraction = 0.2
        self.crossover_prob = 0.01
        self.mutation_prob = 0.2
        self.weight_magnitude_limit = 10000000

        # Save Results
        dummy_env = gym.make(self.env_name)
        self.state_dim = dummy_env.observation_space.shape[0]
        self.action_dim = dummy_env.action_space.shape[0]
        self.hidden_sizes = [400, 300]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_low = float(dummy_env.action_space.low[0])
        self.action_high = float(dummy_env.action_space.high[0])
        self.savefolder = 'results/'
        os.makedirs('results/', exist_ok=True)
        self.aux_folder = self.savefolder + 'Auxiliary/'
        os.makedirs(self.aux_folder, exist_ok=True)


def main():
    args = Parameters()
    save_tag = 'seed_' + str(args.seed)

    frame_tracker = Tracker(args.savefolder, ['score_' + args.env_name + save_tag], '.csv')
    max_tracker = Tracker(args.aux_folder, ['pop_max_score_' + args.env_name, save_tag], '.csv')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = CERL(args)
    print('Running CERL for', args.env_name, 'State_dim:', args.state_dim, ' Action_dim:', args.action_dim)

    start_time = time.time()
    for gen in range(1, 10000000):
        best_score, test_len, all_fitness, all_ep_len, test_mean, test_std, champ_wwid = agent.train(gen, frame_tracker)

        print('Env', args.env_name, 'Gen', gen, 'Frames', agent.total_frames, ' Pop_max/max_ever:', '%.2f' % best_score,
              '/', '%.2f' % agent.best_score, ' Avg:', '%.2f' % frame_tracker.all_tracker[0][1],
              ' Frames/sec:', '%.2f' % (agent.total_frames / (time.time() - start_time)),
              ' Champ_len', '%.2f' % test_len, ' Test_score u/std', str(test_mean), str(test_std),
              'savetag', save_tag)

        if gen % 5 == 0:
            print('Learner Fitness', [str(learner.value) for learner in agent.portfolio],
                  'Sum_stats_resource_allocation', [learner.visit_count for learner in agent.portfolio])
            print('Pop/rollout size', args.pop_size, '/', args.rollout_size, 'Seed', args.seed)
            try:
                print('Best Policy ever genealogy:', agent.genealogy.tree[int(agent.best_policy.wwid.item())].history)
                print('Champ genealogy:', agent.genealogy.tree[champ_wwid].history)
            except:
                pass
            print()

        max_tracker.update([best_score], agent.total_frames)
        if agent.total_frames > 1000000:
            break

        # Save sum stats
        visit_counts = np.array([learner.visit_count for learner in agent.portfolio])
        np.savetxt(args.aux_folder + 'allocation_' + str(args.seed) + save_tag, visit_counts, fmt='%.3f', delimiter=',')

    # Kill all processes
    try:
        for p in agent.task_pipes: p[0].send('TERMINATE')
        for p in agent.test_task_pipes: p[0].send('TERMINATE')
        for p in agent.evo_task_pipes: p[0].send('TERMINATE')

    except:
        pass


if __name__ == '__main__':
    main()
