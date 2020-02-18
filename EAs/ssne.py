import math
from copy import deepcopy

import numpy as np


# Sub-structure based Neuroevolution (SSNE)
class SSNE:
    def __init__(self, args):
        self.gen = 0
        self.args = args
        self.pop_size = self.args.pop_size
        self.rl_sync_pool = []
        self.all_offs = []
        self.rl_res = {'elites': 0.0, 'selects': 0.0, 'discarded': 0.0}
        self.num_rl_syncs = 0.0001
        self.lineage = {0.0 for _ in range(self.pop_size)}
        self.lineage_depth = 10

    @staticmethod
    def selection_tournament(index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        # dummy winner
        winner = 0
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))
        if len(offsprings) % 2 != 0:
            offsprings.append(index_rank[winner])

        return offsprings

    @staticmethod
    def list_argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    @staticmethod
    def regularize_wight(weight, magnitude):
        if weight > magnitude:
            weight = magnitude
        elif weight < - magnitude:
            weight = - magnitude

        return weight

    @staticmethod
    def crossover_inplace(gene1, gene2):
        keys1 = list(gene1.state_dict())
        keys2 = list(gene2.state_dict())

        for key in keys1:
            if key not in keys2:
                continue
            w1 = gene1.state_dict()[key]
            w2 = gene2.state_dict()[key]

            # this random value is used for deciding which weights are replaced
            receiver_choice = np.random.rand()
            # index for replacing
            ind_cr = np.random.randint(0, w1.shape[0] - 1)

            if len(w1.shape) == 2:  # weights
                num_variables = w1.shape[0]
                if int(num_variables * 0.3) < 1:
                    num_cross_overs = np.random.randint(0, int(num_variables * 0.3))
                else:
                    num_cross_overs = 1

                for _ in range(num_cross_overs):
                    if receiver_choice < 0.5:    # replace w1 weights
                        w1[ind_cr, :] = w2[ind_cr, :]
                    else:   # replace w2 weights
                        w2[ind_cr, :] = w1[ind_cr, :]
            elif len(w1.shape) == 1:    # bias or LayerNorm
                if np.random.rand() < 0.8:   # don't crossover
                    continue

                if receiver_choice < 0.5:   # replace w1
                    w1[ind_cr] = w2[ind_cr]
                else:   # replace w2
                    w2[ind_cr] = w1[ind_cr]

    def mutate_inplace(self, gene):
        mut_strength = 0.02
        num_mutation_frac = 0.03
        super_mut_strength = 1.0
        super_mut_prob = 0.1
        reset_prob = super_mut_prob + 0.1

        num_params = len(list(gene.parameters()))
        # mutation probabilities for each parameters
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2

        for i, param in enumerate(gene.parameters()):
            w = param.data

            if len(w.shape) == 2:   # weights
                num_weights = w.shape[0] * w.shape[1]
                ssne_prob = ssne_probabilities[i].item()

                if np.random.rand() < ssne_prob:
                    num_mutations = np.random.randint(0, int(math.ceil(num_mutation_frac * num_weights)))
                    for _ in range(num_mutations):
                        ind_dim1 = np.random.randint(0, w.shape[0] - 1)
                        ind_dim2 = np.random.randint(0, w.shape[1] - 1)
                        random_num = np.random.rand()

                        if random_num < super_mut_prob:
                            w[ind_dim1, ind_dim2] += np.random.normal(0, abs(super_mut_strength * w[ind_dim1, ind_dim2]))
                        elif random_num < reset_prob:
                            w[ind_dim1, ind_dim2] = np.random.normal(0, 0.1)
                        else:
                            w[ind_dim1, ind_dim2] += np.random.normal(0, abs(mut_strength * w[ind_dim1, ind_dim2]))

                        w[ind_dim1, ind_dim2] = self.regularize_wight(
                            w[ind_dim1, ind_dim2],
                            self.args.weight_magnitude_limit
                        )
            elif len(w.shape) == 1:     # bias or LayerNorm
                num_weights = w.shape[0]
                ssne_prob = ssne_probabilities[i].item() * 0.04     # less probability than weights

                if np.random.rand() < ssne_prob:
                    num_mutations = np.random.randint(0, math.ceil(num_mutation_frac * num_weights))
                    for _ in range(num_mutations):
                        ind_dim = np.random.randint(0, w.shape[0] - 1)
                        random_num = np.random.rand()

                        if random_num < super_mut_prob:
                            w[ind_dim] += np.random.normal(0, abs(super_mut_strength * w[ind_dim]))
                        elif random_num < reset_prob:
                            w[ind_dim] = np.random.normal(0, 1)
                        else:
                            w[ind_dim] += np.random.normal(0, abs(mut_strength * w[ind_dim]))

                        w[ind_dim] = self.regularize_wight(w[ind_dim], self.args.weight_magnitude_limit)

    @staticmethod
    def reset_genome(gene):
        for param in (gene.parameters()):
            param.data.copy_(param.data)

    def epoch(self, gen, genealogy, pop, fitness_evals, migration):
        self.gen += 1

        num_elitists = int(self.args.elite_fraction * len(fitness_evals))
        if num_elitists < 2:
            num_elitists = 2

        index_rank = self.list_argsort(fitness_evals)
        index_rank.reverse()
        elitists_index = index_rank[:num_elitists]

        num_offsprings = len(index_rank) - len(elitists_index) - len(migration)
        offsprings = self.selection_tournament(index_rank, num_offsprings, tournament_size=3)

        unselected = []
        for net_index in range(len(pop)):
            if net_index in offsprings or net_index in elitists_index:
                continue
            else:
                unselected.append(net_index)

        np.random.shuffle(unselected)

        # Inheritance step (sync learners to population)
        for policy in migration:
            replaced_one = unselected.pop(0)
            pop[replaced_one] = deepcopy(policy)
            wwid = genealogy.asexual(int(policy.wwid.item()))
            pop[replaced_one].wwid[0] = wwid

        # Elitism
        new_elitists = []
        for i in elitists_index:
            if len(unselected) != 0:
                replaced_one = unselected.pop(0)
            else:
                replaced_one = offsprings.pop(0)
            new_elitists.append(replaced_one)
            pop[replaced_one] = deepcopy(pop[i])
            wwid = genealogy.asexual(int(pop[i].wwid.item()))
            pop[replaced_one].wwid[0] = wwid
            genealogy.elite(wwid, gen)

        # crossover for unselected genes with probability 1
        if len(unselected) % 2 != 0:
            unselected.append(unselected[np.random.randint(0, len(unselected) - 1)])
        for i, j in zip(unselected[0::2], unselected[1::2]):
            off_i = np.random.choice(new_elitists)
            off_j = np.random.choice(offsprings)
            pop[i] = deepcopy(pop[off_i])
            pop[j] = deepcopy(pop[off_j])
            self.crossover_inplace(pop[i], pop[j])
            wwid1 = genealogy.crossover(gen)
            wwid2 = genealogy.crossover(gen)
            pop[i].wwid[0] = wwid1
            pop[j].wwid[0] = wwid2

        # crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if np.random.rand() < self.args.crossover_prob:
                self.crossover_inplace(pop[i], pop[j])
                wwid1 = genealogy.crossover(gen)
                wwid2 = genealogy.crossover(gen)
                pop[i].wwid[0] = wwid1
                pop[j].wwid[0] = wwid2

        # mutate all genes in the population except the new elitists
        for net_index in range(len(pop)):
            if net_index not in new_elitists:
                if np.random.rand() < self.args.mutation_prob:
                    self.mutate_inplace(pop[net_index])
                    genealogy.mutation(int(pop[net_index].wwid.item()), gen)

        self.all_offs[:] = offsprings[:]
