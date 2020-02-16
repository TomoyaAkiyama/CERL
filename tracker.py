import os

import numpy as np


class Tracker:
    def __init__(self, save_folder, vars_string, project_string):
        self.vars_string = vars_string
        self.project_string = project_string
        self.folder_name = save_folder
        self.all_tracker = [[[], 0.0, []] for _ in vars_string]
        self.counter = 0
        self.conv_size = 1
        os.makedirs(self.folder_name, exist_ok=True)

    def update(self, updates, generation):

        self.counter += 1
        for update, var in zip(updates, self.all_tracker):
            if update is None:
                continue
            var[0].append(updates)

        for var in self.all_tracker:
            if len(var[0]) > self.conv_size:
                var[0].pop(0)

        for var in self.all_tracker:
            if len(var[0]) == 0:
                continue
            var[1] = np.array(var[0]).mean()

        if self.counter % 1 == 0:
            for i, var in enumerate(self.all_tracker):
                if len(var[0]) == 0:
                    continue
                var[2].append(np.array([generation, var[1]]))
                filename = self.folder_name + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')
