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


def plot_average_test_scores(portfolio_id, env_name, seeds, x_min, x_max, y_min, y_max):
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    all_test_scores = []
    time_steps = []
    for seed in seeds:
        save_dir = os.path.join(parent_dir, 'results', portfolio_id, env_name, 'seed{}'.format(seed))
        file_name = 'test_scores.txt'
        file_path = os.path.join(save_dir, file_name)
        test_data = np.loadtxt(file_path)
        time_steps = test_data[:, 0]
        all_test_scores.append(test_data[:, 1])

    average_test_scores = np.mean(all_test_scores, axis=0)
    std = np.std(all_test_scores, axis=0)

    plt.figure()
    plt.plot(time_steps, average_test_scores)
    plt.fill_between(time_steps, average_test_scores - std, average_test_scores + std, facecolor='c', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Average Episode Rewards')
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig_name = 'average_test_scores.png'
    save_dir = os.path.join(parent_dir, 'results', portfolio_id, env_name)
    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path)


if __name__ == '__main__':
    portfolio_id = 'portfolio1'
    env_name = 'HalfCheetah-v2'
    seeds = range(1, 4)

    x_min = 0
    x_max = 1000000
    y_min = -2000
    y_max = 8000

    for seed in seeds:
        plot_test_scores(portfolio_id, env_name, seed, x_min, x_max, y_min, y_max)
    plot_average_test_scores(portfolio_id, env_name, seeds, x_min, x_max, y_min, y_max)