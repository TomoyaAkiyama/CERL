import copy
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from models.deterministic_policy import DeterministicPolicy
from models.double_q import DoubleQ
from models.model_utils import init_weights


class TD3:
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_sizes,
            device,
            wwid,
            gamma=0.99,
            tau=0.005,
            lr=3e-4,
            policy_noise=0.2,
            noise_clip=0.5,
            exploration_noise=0.1,
            policy_freq=2
    ):

        self.actor = DeterministicPolicy(state_dim, action_dim, hidden_sizes, wwid).to(device)
        self.actor.apply(init_weights)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=lr)
        self.critic = DoubleQ(state_dim, action_dim, hidden_sizes).to(device)
        self.critic.apply(init_weights)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr)

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.rollout_actor = TD3RolloutActor(state_dim, action_dim, hidden_sizes, wwid, exploration_noise)
        self.sync_rollout_actor()

        self.iteration_num = 0

    def train(self, replay_buffer, batch_size=256):
        self.iteration_num += 1

        st, nx_st, ac, rw, mask = replay_buffer.sample(batch_size)
        st = st.to(self.device)
        nx_st = nx_st.to(self.device)
        ac = ac.to(self.device)
        rw = rw.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(ac) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            nx_ac = (self.target_actor.forward(nx_st) + noise).clamp(0, 1)

            target_q1, target_q2 = self.target_critic.forward(nx_st, nx_ac)
            min_q = torch.min(target_q1, target_q2)
            target_q = rw + mask * self.gamma * min_q

        q1, q2 = self.critic.forward(st, ac)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.iteration_num % self.policy_freq == 0:
            actor_loss = - self.critic.q1(st, self.actor.forward(st)).mean()
            self.actor.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.sync_rollout_actor()

    def sync_rollout_actor(self):
        for param, target_param in zip(self.actor.parameters(), self.rollout_actor.parameters()):
            target_param.data.copy_(param.data.cpu())

    def save(self, path):
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        torch.save(self.target_critic.state_dict(), os.path.join(path, 'target_critic.pth'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, 'critic_optimizer.pth'))

        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.target_actor.state_dict(), os.path.join(path, 'target_actor.pth'))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, 'actor_optimizer.pth'))

    def load(self, path):
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))
        self.target_critic.load_state_dict(torch.load(os.path.join(path, 'target_critic.pth')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, 'critic_optimizer.pth')))

        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
        self.target_actor.load_state_dict(torch.load(os.path.join(path, 'target_actor.pth')))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(path, 'actor_optimizer.pth')))
        self.sync_rollout_actor()


class TD3RolloutActor:
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_sizes,
            wwid,
            exploration_noise
    ):
        self.actor = DeterministicPolicy(state_dim, action_dim, hidden_sizes, wwid).eval()
        self.exploration_noise = exploration_noise

    def select_action(self, state):
        action = self.actor.forward(state)
        noise = torch.randn_like(action) * self.exploration_noise
        action = action + noise
        return action.cpu().detach().numpy().flatten()

    def deterministic_action(self, state):
        action = self.actor.forward(state)
        return action.cpu().detach().numpy().flatten()

    def parameters(self):
        return self.actor.parameters()
