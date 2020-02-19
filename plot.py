import os

import numpy as np
import matplotlib.pyplot as plt


def plot_test_scores(portfolio_id, env_name, seed, x_min, x_max, y_min, y_max):
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(parent_dir, 'results', portfolio_id, env_name, 'seed{}'.format(seed))
    file_name = 'test_scores.txt'
    file_path = os.path.join(save_dir, file_name)
    test_data = np.loadtxt(file_path)
    time_steps = test_data[:, 0]
    test_scores = test_data[:, 1]

    plt.figure()
    plt.plot(time_steps, test_scores)
    plt.xlabel('Time Step')
    plt.ylabel('Average Episode Rewards')
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig_name = 'test_scores.png'
    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path)


if __name__ == '__main__':
    portfolio_id = 'portfolio1'
    env_name = 'HalfCheetah-v2'
    seed = 1

    x_min = 0
    x_max = 1000000
    y_min = -2000
    y_max = 8000

    plot_test_scores(portfolio_id, env_name, seed, x_min, x_max, y_min, y_max)
