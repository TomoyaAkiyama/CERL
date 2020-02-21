from copy import deepcopy
import threading
from collections import Counter

import numpy as np
from torch.multiprocessing import Manager, Pipe, Process
import gym

from EAs import SSNE
from genealogy import Genealogy
from replay_buffer import ReplayBuffer
from models.utils import init_policy
from portfolio import init_portfolio
from rollout_worker import rollout_worker
from ucb import ucb


def env_parse(env_name):
    dummy_env = gym.make(env_name)
    state_dim = sum(list(dummy_env.observation_space.shape))
    action_dim = sum(list(dummy_env.action_space.shape))
    return state_dim, action_dim


class CERL:
    def __init__(
            self,
            env_name,
            rollout_size=10,
            pop_size=10,
            portfolio_id='portfolio1',
            policy_type='Deterministic',
            hidden_sizes=None,
            use_cuda=False,
            capacity=int(1e6),
            batch_size=256,
            ucb_coefficient=0.9,
            elite_fraction=0.2,
            cross_prob=0.01,
            cross_fraction=0.3,
            bias_cross_prob=0.2,
            mutation_prob=0.2,
            mut_strength=0.02,
            mut_fraction=0.03,
            super_mut_prob=0.1,
            reset_prob=0.2,
    ):
        if hidden_sizes is None:
            hidden_sizes = [400, 300]

        ea_kwargs = {
            'elite_fraction': elite_fraction,
            'cross_prob': cross_prob,
            'cross_fraction': cross_fraction,
            'bias_cross_prob': bias_cross_prob,
            'mutation_prob': mutation_prob,
            'mut_strength': mut_strength,
            'mut_fraction': mut_fraction,
            'super_mut_prob': super_mut_prob,
            'reset_prob': reset_prob,
        }
        self.EA = SSNE(**ea_kwargs)

        self.manager = Manager()
        self.genealogy = Genealogy()

        self.rollout_size = rollout_size
        self.pop_size = pop_size

        self.batch_size = batch_size
        self.ucb_coefficient = ucb_coefficient

        state_dim, action_dim = env_parse(env_name)

        # policies for EA's rollout
        self.population = self.manager.list()
        for i in range(pop_size):
            wwid = self.genealogy.new_id('EA_{}'.format(i))
            policy = init_policy(state_dim, action_dim, hidden_sizes, wwid, policy_type).eval()
            self.population.append(policy)
        self.best_policy = init_policy(state_dim, action_dim, hidden_sizes, -1, policy_type).eval()

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, capacity)

        self.portfolio = init_portfolio(state_dim, action_dim, hidden_sizes, use_cuda, self.genealogy, portfolio_id)
        # policies for learners' rollout
        self.rollout_bucket = self.manager.list()
        for learner in self.portfolio:
            self.rollout_bucket.append(learner.algo.rollout_actor)

        # Evolutionary population Rollout workers
        self.evo_task_pipes = []
        self.evo_result_pipes = []
        self.evo_workers = []
        for index in range(pop_size):
            task_pipe = Pipe()
            result_pipe = Pipe()
            worker_args = (index, task_pipe[1], result_pipe[0], False, self.population, env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.evo_task_pipes.append(task_pipe)
            self.evo_result_pipes.append(result_pipe)
            self.evo_workers.append(worker)
            worker.start()

        # Learner rollout workers
        self.learner_task_pipes = []
        self.learner_result_pipes = []
        self.learner_workers = []
        for index in range(rollout_size):
            task_pipe = Pipe()
            result_pipe = Pipe()
            worker_args = (index, task_pipe[1], result_pipe[0], True, self.rollout_bucket, env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.learner_task_pipes.append(task_pipe)
            self.learner_result_pipes.append(result_pipe)
            self.learner_workers.append(worker)
            worker.start()

        # test bucket
        self.test_bucket = self.manager.list()
        policy = init_policy(state_dim, action_dim, hidden_sizes, -1, policy_type)
        self.test_bucket.append(policy)
        self.test_task_pipes = []
        self.test_result_pipes = []
        self.test_workers = []
        for index in range(10):
            task_pipe = Pipe()
            result_pipe = Pipe()
            worker_args = (index, task_pipe[1], result_pipe[0], False, self.test_bucket, env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.test_task_pipes.append(task_pipe)
            self.test_result_pipes.append(result_pipe)
            self.test_workers.append(worker)
            worker.start()

        self.allocation = []
        for i in range(rollout_size):
            self.allocation.append(i % len(self.portfolio))

        self.best_score = - float('inf')
        self.gen_frames = 0
        self.total_frames = 0

    # receive EA's rollout
    def receive_ea_rollout(self, all_transitions):
        pop_fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
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
        for i in range(self.rollout_size):
            entry = self.learner_result_pipes[i][1].recv()
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
        for index in range(self.pop_size):
            self.evo_task_pipes[index][0].send(index)

        # learners' rollout
        for rollout_index, learner_index in enumerate(self.allocation):
            self.learner_task_pipes[rollout_index][0].send(learner_index)

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
        if gen % 5 == 1:
            for pipe in self.test_task_pipes:
                pipe[0].send(0)

        # add all transitions to replay buffer
        self.replay_buffer.add_transitions(all_transitions)

        # update learners' parameters
        if len(self.replay_buffer) > self.batch_size * 10:
            threads = []
            for learner in self.portfolio:
                threads.append(
                    threading.Thread(target=learner.update_parameters,
                                     args=(self.replay_buffer, self.batch_size, int(self.gen_frames)))
                )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.gen_frames = 0

        test_scores = []
        for pipe in self.test_result_pipes:
            entry = pipe[1].recv()
            test_scores.append(entry[1])
        test_mean = np.mean(test_scores)
        test_std = np.std(test_scores)
        logger.add_test_score(self.total_frames, test_mean.item())

        # EA step
        if gen % 5 == 0:
            migration = [deepcopy(rollout_actor.actor) for rollout_actor in self.rollout_bucket]
            self.EA.epoch(gen, self.genealogy, self.population, pop_fitness.tolist(), migration)
        else:
            self.EA.epoch(gen, self.genealogy, self.population, pop_fitness.tolist(), [])

        # allocate learners' rollout according to ucb score
        self.allocation = ucb(len(self.allocation), self.portfolio, self.ucb_coefficient)

        champ_wwid = int(self.population[champ_index].wwid.item())

        return pop_fitness, learner_fitness, alloc_count, test_mean, test_std, champ_wwid
