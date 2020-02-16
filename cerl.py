from copy import deepcopy
import threading

import numpy as np
from torch.multiprocessing import Manager, Pipe, Process

from EAs import SSNE
from genealogy import Genealogy
from replay_buffer import ReplayBuffer
from models.utils import init_policy
from portfolio import init_portfolio
from rollout_worker import rollout_worker
from ucb import ucb


class CERL:
    def __init__(self, args):
        self.args = args
        self.EA = SSNE(self.args)

        self.manager = Manager()
        self.genealogy = Genealogy()

        # policies for EA's rollout
        self.population = self.manager.list()
        for _ in range(args.pop_size):
            wwid = self.genealogy.new_id('EA')
            policy = init_policy(args.state_dim, args.action_dim, args.hidden_sizes, wwid, args.policy_type).eval()
            self.population.append(policy)
        self.best_policy = init_policy(args.state_dim, args.action_dim, args.hidden_sizes, -1, args.policy_type).eval()

        self.replay_buffer = ReplayBuffer(args.state_dim, args.action_dim, args.device, args.capacity)

        self.portfolio = init_portfolio(args.state_dim, args.action_dim, args.device, self.genealogy)
        # policies for learners' rollout
        self.rollout_bucket = self.manager.list()
        for learner in self.portfolio:
            self.rollout_bucket.append(learner)

        self.tmp_buffer = self.replay_buffer.tmp_buffer

        # Evolutionary population Rollout workers
        self.evo_task_pipes = []
        self.evo_result_pipes = []
        self.evo_workers = []
        for index in range(args.pop_size):
            task_pipe = Pipe()
            result_pipe = Pipe()
            worker_args = (index, task_pipe[1], result_pipe[0], False, self.tmp_buffer, self.population, args.env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.evo_task_pipes.append(task_pipe)
            self.evo_result_pipes.append(result_pipe)
            self.evo_workers.append(worker)
            worker.start()
        self.evo_flag = [True] * args.pop_size

        # Learner rollout workers
        self.task_pipes = []
        self.result_pipes = []
        self.workers = []
        for index in range(args.rollout_size):
            task_pipe = Pipe()
            result_pipe = Pipe()
            worker_args = (index, task_pipe[1], result_pipe[0], True, self.tmp_buffer, self.rollout_bucket, args.env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.task_pipes.append(task_pipe)
            self.result_pipes.append(result_pipe)
            self.workers.append(worker)
            worker.start()
        self.roll_flag = [True] * args.rollout_size

        # test bucket
        self.test_bucket = self.manager.list()
        policy = init_policy(args.state_dim, args.action_dim, args.hidden_sizes, -1, args.policy_type)
        self.test_bucket.append(policy)
        self.test_task_pipes = []
        self.test_result_pipes = []
        self.test_workers = []
        for index in range(args.test_size):
            task_pipe = Pipe()
            result_pipe = Pipe()
            worker_args = (index, task_pipe[1], result_pipe[0], False, None, self.test_bucket, args.env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.test_task_pipes.append(task_pipe)
            self.test_result_pipes.append(result_pipe)
            self.test_workers.append(worker)
            worker.start()
        self.test_flag = False

        self.allocation = []
        for i in range(args.rollout_size):
            self.allocation.append(i % len(self.portfolio))

        self.best_score = 0.0
        self.gen_frames = 0
        self.total_frames = 0
        self.best_shaped_score = None
        self.test_score = None
        self.test_std = None

    def train(self, gen, frame_tracker):
        # EA's rollout
        for index, policy in enumerate(self.population):
            if self.evo_flag[index]:
                self.evo_task_pipes[index][0].send(index)
                self.evo_flag[index] = False

        # # copy all learners' policy to cpu rollout_bucket
        # for i, learner in enumerate(self.portfolio):
        #     self.rollout_bucket[i] = deepcopy(learner.algo.actor).cpu().eval()

        # learners' rollout
        for rollout_index, learner_index in enumerate(self.allocation):
            if self.roll_flag[rollout_index]:
                self.task_pipes[rollout_index][0].send(learner_index)
                self.roll_flag[rollout_index] = False

        # test policy
        if gen % 5 == 0:
            self.test_flag = True
            for pipe in self.test_task_pipes:
                pipe[0].send(0)

        # update learners' parameters
        if len(self.replay_buffer) > self.args.batch_size * 10:
            self.replay_buffer.refresh()
            for learner in self.portfolio:
                thread = threading.Thread(target=learner.update_parameters,
                                          args=(self.replay_buffer, self.args.batch_size, int(self.gen_frames)))
                thread.start()
                thread.join()
                self.gen_frames = 0

        # receive EA's rollout
        all_net_indices = []
        all_fitness = []
        all_ep_lens = []
        while True:
            for i in range(self.args.pop_size):
                if self.evo_result_pipes[i][1].poll():
                    entry = self.evo_result_pipes[i][1].recv()
                    all_net_indices.append(entry[0])
                    all_fitness.append(entry[1])
                    all_ep_lens.append(entry[2])
                    self.gen_frames += entry[2]
                    self.total_frames += entry[2]
                    self.evo_flag[i] = True

            # soft join (50%)
            # self.args.asynch_frac = 1 ならいらない気がする. フラグ管理も同時にいらなくなる？
            if len(all_fitness) / self.args.pop_size >= self.args.asynch_frac:
                break

        # receive learners' rollout
        for i in range(self.args.rollout_size):
            entry = self.result_pipes[i][1].recv()
            learner_id = entry[0]
            fitness = entry[1]
            num_frames = entry[2]
            self.portfolio[learner_id].update_stats(fitness, num_frames)

            self.gen_frames += num_frames
            self.total_frames += num_frames
            if fitness > self.best_score:
                self.best_score = fitness
                # これがいると思う
                # self.best_policy = deepcopy(self.portfolio[learner_id].algo.actor).cpu().eval()

            self.roll_flag[i] = True

        self.replay_buffer.refresh()

        # champion policy
        max_fit = max(all_fitness)
        champ_index = all_net_indices[all_fitness.index(max_fit)]
        self.test_bucket = deepcopy(self.population[champ_index])
        if max_fit > self.best_score:
            self.best_score = max_fit
            self.best_policy = deepcopy(self.population[champ_index])

        # test score
        if self.test_flag:
            self.test_flag = False
            test_scores = []
            for pipe in self.test_result_pipes:
                entry = pipe[1].recv()
                test_scores.append(entry[1])
            test_mean = np.mean(test_scores)
            test_std = np.std(test_scores)

            frame_tracker.update([test_mean], self.total_frames)
        else:
            test_mean = None
            test_std = None

        # EA step
        if gen % 5 == 0:
            migration = [deepcopy(learner.algo.actor).cpu().eval() for learner in self.portfolio]
            self.EA.epoch(gen, self.genealogy, self.population, all_net_indices, all_fitness, migration)
        else:
            self.EA.epoch(gen, self.genealogy, self.population, all_net_indices, all_fitness, [])

        self.allocation = ucb(len(self.allocation), self.portfolio, self.args.ucb_coefficient)

        champ_len = all_ep_lens[all_fitness.index(max_fit)]
        champ_wwid = int(self.population[champ_index].wwid.item())

        return max_fit, champ_len, all_fitness, all_ep_lens, test_mean, test_std, champ_wwid



