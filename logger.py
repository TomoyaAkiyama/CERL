import os

import numpy as np


class Logger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # learners' info
        self.learner_fitnesses = []
        self.learner_values = []
        self.allocation_counts = []

        # population's info
        self.population_fitnesses = []
        self.test_scores = []

    def add_learner_value(self, total_frame, portfolio):
        values = [learner.value for learner in portfolio]
        item = [total_frame] + values
        self.learner_values.append(item)

    def add_allocation_count(self, total_frame, allocation_count):
        item = [total_frame] + allocation_count
        self.allocation_counts.append(item)

    def add_fitness(self, total_frame, population_fitness, learner_fitness):
        item = [total_frame] + population_fitness
        self.population_fitnesses.append(item)

        item = [total_frame] + learner_fitness
        self.learner_fitnesses.append(item)

    def add_test_score(self, total_frame, test_score):
        item = [total_frame] + test_score
        self.test_scores.append(item)

    def save(self):
        file_name = 'learner_values.txt'
        file_path = os.path.join(self.save_dir, file_name)
        np.savetxt(file_path, self.learner_values)

        file_name = 'allocation_counts.txt'
        file_path = os.path.join(self.save_dir, file_name)
        np.savetxt(file_path, self.allocation_counts)

        file_name = 'population_fitnesses.txt'
        file_path = os.path.join(self.save_dir, file_name)
        np.savetxt(file_path, self.population_fitnesses)

        file_name = 'learner_fitnesses.txt'
        file_path = os.path.join(self.save_dir, file_name)
        np.savetxt(file_path, self.learner_fitnesses)

        file_name = 'test_scores.txt'
        file_path = os.path.join(self.save_dir, file_name)
        np.savetxt(file_path, self.test_scores)
