import numpy as np


def ucb(allocation_size, portfolio, coefficient):
    values = np.array([learner.value for learner in portfolio])
    values = values - np.min(values)
    values = values / (values.sum() + 0.1)

    visit_counts = [learner.visit_count for learner in portfolio]
    total_visit = np.sum(visit_counts)

    ucb_scores = values + coefficient * np.sqrt(np.log(total_visit) / visit_counts)

    allocation = roulette_wheel(ucb_scores, allocation_size)

    return allocation.tolist()


def roulette_wheel(probs, num_samples):
    probs = probs - np.min(probs) + np.abs(np.min(probs))

    if probs.sum().item() != 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(len(probs))

    out = np.random.choice(len(probs), size=num_samples, p=probs)

    print('UCB_prob_mass', ["%.2f" %i for i in probs])
    print('Allocation', out)
    print()

    return out
