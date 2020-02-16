import torch

from env_wrapper import EnvWrapper


def rollout_worker(index, task_pipe, result_pipe, explore, tmp_buffer, model_bucket, env_name):
    env = EnvWrapper(env_name)
    env.seed(index)

    while True:
        identifier = task_pipe.recv()
        if identifier == 'TERMINATE':
            exit(0)

        agent = model_bucket[identifier]

        fitness = 0.0
        total_frame = 0
        ep_frame = 0
        state = env.reset()
        rollout_transition = []

        while True:
            if explore:
                action = agent.select_action(torch.tensor(state.reshape(1, -1), dtype=torch.float))
            else:
                action = agent.deterministic_action(torch.tensor(state.reshape(1, -1), dtype=torch.float))

            next_state, reward, done, info = env.step(action)
            fitness += reward
            total_frame += 1
            ep_frame += 1

            if tmp_buffer is not None:
                done_buffer = done if ep_frame < env.env._max_episode_steps else False

                rollout_transition.append({
                    'state': state,
                    'next_state': next_state,
                    'action': action,
                    'reward': reward,
                    'mask': float(not done_buffer)
                })
            state = next_state

            if done:
                for entry in rollout_transition:
                    tmp_buffer.append(entry)
                break

        result_pipe.send([identifier, fitness, total_frame])
