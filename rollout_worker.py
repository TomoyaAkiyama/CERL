import torch

from env_wrapper import EnvWrapper


def rollout_worker(index, task_pipe, result_pipe, explore, model_bucket, env_name):
    env = EnvWrapper(env_name)
    env.seed(index)

    while True:
        identifier = task_pipe.recv()
        if identifier == 'TERMINATE':
            exit(0)

        policy = model_bucket[identifier]

        fitness = 0.0
        total_frame = 0
        state = env.reset()
        done = False
        rollout_transition = []

        while not done:
            if explore:
                action = policy.select_action(torch.tensor(state.reshape(1, -1), dtype=torch.float))
            else:
                action = policy.deterministic_action(torch.tensor(state.reshape(1, -1), dtype=torch.float))

            next_state, reward, done, info = env.step(action)
            fitness += reward
            total_frame += 1

            done_buffer = done if total_frame < env.unwrapped()._max_episode_steps else False

            rollout_transition.append({
                'state': state,
                'next_state': next_state,
                'action': action,
                'reward': reward,
                'mask': float(not done_buffer)
            })
            state = next_state

        result_pipe.send([identifier, fitness, rollout_transition])
