import os
import copy

import torch
from torch.optim import Adam
import torch.nn.functional as F

from models.double_q import DoubleQ
from models.gaussian_policy import GaussianPolicy


class SAC:
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_sizes,
            device,
            wwid,
            gamma=0.99,
            tau=0.005,
            lr=3e-4
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = - action_dim

        self.critic = DoubleQ(state_dim, action_dim, hidden_sizes).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_sizes, wwid).to(device)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1).to(device), requires_grad=True)
        self.alpha = self.log_alpha.exp()

        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)

    def select_action(self, state):
        state = state.to(self.device)
        action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy().flatten()

    def deterministic_action(self, state):
        state = state.to(self.device)
        action, _ = self.actor.forward(state)
        action = torch.tanh(action)
        return action.cpu().detach().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        st, nx_st, ac, rw, mask = replay_buffer.random_batch(batch_size)

        with torch.no_grad():
            nx_ac, nx_ac_log_prob = self.actor.sample(nx_st)
            nx_q1, nx_q2 = self.target_critic.forward(nx_st, nx_ac)
            nx_q_min = torch.min(nx_q1, nx_q2)
            nx_values = nx_q_min - self.alpha * nx_ac_log_prob
            q_target = rw + mask * self.gamma * nx_values

        new_ac, new_ac_log_prob = self.actor.sample(st)
        q1, q2 = self.critic.forward(st, ac)

        new_q1, new_q2 = self.critic.forward(st, new_ac)
        new_q = torch.min(new_q1, new_q2)

        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        actor_loss = ((self.alpha * new_ac_log_prob) - new_q).mean()
        alpha_loss = - (self.log_alpha * (new_ac_log_prob.detach() + self.target_entropy)).mean()

        self.critic_optimizer.zero_grad()
        q1_loss.backward()
        self.critic_optimizer.step()

        self.critic_optimizer.zero_grad()
        q2_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        torch.save(self.target_critic.state_dict(), os.path.join(path, 'target_critic.pth'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, 'critic_optimizer.pth'))

        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, 'actor_optimizer.pth'))

        torch.save(self.log_alpha.data, os.path.join(path, 'log_alpha.pth'))

    def load(self, path):
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))
        self.target_critic.load_state_dict(torch.load(os.path.join(path, 'target_critic.pth')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, 'critic_optimizer.pth')))

        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(path, 'actor_optimizer.pth')))

        self.log_alpha = torch.nn.Parameter(torch.load(os.path.join(path, 'log_alpha.pth')), requires_grad=True)
        self.alpha = self.log_alpha.exp()


