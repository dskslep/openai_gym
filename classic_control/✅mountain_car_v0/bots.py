from agent_templates.simple_td import QLearning


class QBot(QLearning):
    def __init__(self, env, grid_size, epsilon, gamma, step_size, start_values, step_planning,
                 episode_planning):
        num_actions = env.action_space.n
        state_size = env.observation_space.shape[0]

        self.high = env.observation_space.high

        self.low = env.observation_space.low
        self.grid_size = grid_size

        super().__init__(epsilon, num_actions, gamma, step_size, state_size, start_values, step_planning,
                         episode_planning)

    def observation_to_state(self, observation):
        normed_obs = (observation - self.low) / (self.high - self.low)
        res = [int(x * self.grid_size) for x in normed_obs]
        return tuple(res)
