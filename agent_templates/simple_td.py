from abc import ABC

from agent_templates.base import SimpleTD

import numpy as np


class QLearning(SimpleTD, ABC):

    def target_val(self, reward, state, done):
        return reward + (1 - done) * self.gamma * np.max(self.state_values(state))


class ExpectedSarsa(SimpleTD, ABC):

    def target_val(self, reward, state, done):
        state_values = self.state_values(state)
        val = reward + (1 - done) * self.gamma * (
            (1 - self.epsilon) * np.max(state_values) + self.epsilon * np.mean(state_values)
        )
        return val


