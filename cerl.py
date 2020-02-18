from copy import deepcopy
import threading
from collections import Counter

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
        for i in range(args.pop_size):
            wwid = self.genealogy.new_id('EA_{}'.format(i))
            policy = init_policy(args.state_dim, args.action_dim, args.hidden_sizes, wwid, args.policy_type).eval()
            self.population.append(policy)
        self.best_policy = init_policy(args.state_dim, args.action_dim, args.hidden_sizes, -1, args.policy_type).eval()

        self.replay_buffer = ReplayBuffer(args.state_dim, args.action_dim, args.capacity)

        self.portfolio = init_portfolio(args.state_dim, args.action_dim, args.use_cuda, self.genealogy)
        # policies for learners' rollout
        self.rollout_bucket = self.manager.list()
        for learner in self.portfolio:
            self.rollout_bucket.append(learner.algo.rollout_actor)

        # Evolutionary population Rollout workers
        self.evo_task_pipes = []
        self.evo_result_pipes = []
        self.evo_workers = []
        for index in range(args.pop_size):
            task_pipe = Pipe()
            result_pipe = Pipe()
            worker_args = (index, task_pipe[1], result_pipe[0], False, self.population, args.env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.evo_task_pipes.append(task_pipe)
            self.evo_result_pipes.append(result_pipe)
            self.evo_workers.append(worker)
            worker.start()

        # Learner rollout workers
        self.task_pipes = []
        self.result_pipes = []
        self.workers = []
        for index in range(args.rollout_size):
            task_pipe = Pipe()
            result_pipe = Pipe()
            worker_args = (index, task_pipe[1], result_pipe[0], True, self.rollout_bucket, args.env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.task_pipes.append(task_pipe)
            self.result_pipes.append(result_pipe)
            self.workers.append(worker)
            worker.start()

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
            worker_args = (index, task_pipe[1], result_pipe[0], False, self.test_bucket, args.env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.test_task_pipes.append(task_pipe)
            self.test_result_pipes.append(result_pipe)
            self.test_workers.append(worker)
            worker.start()

        self.allocation = []
        for i in range(args.rollout_size):
            self.allocation.append(i % len(self.portfolio))

        self.best_score = - float('inf')
        self.gen_frames = 0
        self.total_frames = 0

    # receive EA's rollout
    def receive_ea_rollout(self, all_transitions):
        pop_fitness = np.zeros(self.args.pop_size)
        for i in range(self.args.pop_size):
            entry = self.evo_result_pipes[i][1].recv()
            net_index = entry[0]
            fitness = entry[1]
            transitions = entry[2]
            pop_fitness[i] = fitness
            all_transitions.extend(transitions)

            self.gen_frames += len(transitions)
            self.total_frames += len(transitions)
            if fitness > self.best_score:
                self.best_score = fitness
                self.best_policy = deepcopy(self.population[net_index])
        return all_transitions, pop_fitness

    # receive learners' rollout
    def receive_learner_rollout(self, all_transitions, alloc_count):
        learner_fitness = np.zeros(len(self.portfolio))
        for i in range(self.args.rollout_size):
            entry = self.result_pipes[i][1].recv()
            learner_id = entry[0]
            fitness = entry[1]
            transitions = entry[2]
            num_frames = len(transitions)
            learner_fitness[learner_id] += fitness
            all_transitions.extend(transitions)

            self.gen_frames += num_frames
            self.total_frames += num_frames
            self.portfolio[learner_id].update_stats(fitness, num_frames)
            if fitness > self.best_score:
                self.best_score = fitness
                self.best_policy = deepcopy(self.portfolio[learner_id].algo.rollout_actor.actor)
        # calc average_fitness
        denom = np.array(alloc_count, dtype=np.float)
        denom[denom == 0] = np.nan
        learner_fitness = learner_fitness / denom

        return all_transitions, learner_fitness

    def train(self, gen, logger):
        # EA's rollout
        for index in range(self.args.pop_size):
            self.evo_task_pipes[index][0].send(index)

        # learners' rollout
        for rollout_index, learner_index in enumerate(self.allocation):
            self.task_pipes[rollout_index][0].send(learner_index)

        # logging allocation count
        alloc_counter = Counter(self.allocation)
        alloc_count = [alloc_counter[i] if i in alloc_counter else 0 for i in range(len(self.portfolio))]
        logger.add_allocation_count(self.total_frames, alloc_count)

        # receive all transitions
        all_transitions = []
        all_transitions, pop_fitness = self.receive_ea_rollout(all_transitions)
        all_transitions, learner_fitness = self.receive_learner_rollout(all_transitions, alloc_count)

        # logging population fitness and learners fitness
        logger.add_fitness(self.total_frames, pop_fitness.tolist(), learner_fitness.tolist())
        # logging learners value
        logger.add_learner_value(self.total_frames, self.portfolio)

        # test champ policy in the population
        champ_index = pop_fitness.argmax()
        self.test_bucket[0] = deepcopy(self.population[champ_index])
        test_frame = self.total_frames - self.gen_frames
        if gen % 5 == 1:
            for pipe in self.test_task_pipes:
                pipe[0].send(0)

        # update learners' parameters
        if len(self.replay_buffer) > self.args.batch_size * 10:
            threads = []
            for learner in self.portfolio:
                threads.append(
                    threading.Thread(target=learner.update_parameters,
                                     args=(self.replay_buffer, self.args.batch_size, int(self.gen_frames)))
                )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.gen_frames = 0

        # add all transitions to replay buffer
        self.replay_buffer.add_transitions(all_transitions)

        if gen % 5 == 1:
            test_scores = []
            for pipe in self.test_result_pipes:
                entry = pipe[1].recv()
                test_scores.append(entry[1])
            test_mean = np.mean(test_scores)
            test_std = np.std(test_scores)

            logger.add_test_score(test_frame, test_mean.item())
        else:
            test_mean = None
            test_std = None

        # EA step
        if gen % 5 == 0:
            migration = [deepcopy(rollout_actor.actor) for rollout_actor in self.rollout_bucket]
            self.EA.epoch(gen, self.genealogy, self.population, pop_fitness.tolist(), migration)
        else:
            self.EA.epoch(gen, self.genealogy, self.population, pop_fitness.tolist(), [])

        # allocate learners' rollout according to ucb score
        self.allocation = ucb(len(self.allocation), self.portfolio, self.args.ucb_coefficient)

        champ_wwid = int(self.population[champ_index].wwid.item())

        return pop_fitness, learner_fitness, alloc_count, test_mean, test_std, champ_wwid
