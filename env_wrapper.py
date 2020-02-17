import gym


class EnvWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.action_low = float(self.env.action_space.low[0])
        self.action_high = float(self.env.action_space.high[0])

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = (self.action_high + self.action_low) / 2. + action * (self.action_high - self.action_low) / 2.
        return self.env.step(action)

    def render(self):
        self.render()

    def seed(self, seed):
        self.env.seed(seed)

    def unwrapped(self):
        return self.env
