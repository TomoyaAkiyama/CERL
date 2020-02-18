import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, capacity):
        self.capacity = capacity

        self.state = np.zeros((capacity, state_dim))
        self.next_state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.reward = np.zeros((capacity, 1))
        self.mask = np.ones((capacity, 1))

        self.device = device
        self._step = 0
        self._size = 0

    def __len__(self):
        return self._size

    def add_transitions(self, transitions):
        for transition in transitions:
            self.state[self._step] = transition['state']
            self.next_state[self._step] = transition['next_state']
            self.action[self._step] = transition['action']
            self.reward[self._step] = transition['reward']
            self.mask[self._step] = transition['mask']

            self._step = (self._step + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self._size, size=batch_size)

        state_batch = self.state[indices]
        next_state_batch = self.next_state[indices]
        actions_batch = self.action[indices]
        rewards_batch = self.reward[indices]
        masks_batch = self.mask[indices]

        return (
            torch.tensor(state_batch, dtype=torch.float).to(self.device),
            torch.tensor(next_state_batch, dtype=torch.float).to(self.device),
            torch.tensor(actions_batch, dtype=torch.float).to(self.device),
            torch.tensor(rewards_batch, dtype=torch.float).to(self.device),
            torch.tensor(masks_batch, dtype=torch.float).to(self.device)
        )
