import json
from abc import ABC, abstractmethod
from collections import defaultdict
from random import choice

import numpy as np


class Greedy(ABC):
    def __init__(self, epsilon, num_actions):
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.prev_action = None
        self.prev_state = None

    def start(self, observation, rand):
        state = self.observation_to_state(observation)

        if rand:
            action = np.random.choice(range(self.num_actions))
            self.prev_action = action
            self.prev_state = state
        else:
            action = self.policy_action(observation)

        return action

    @abstractmethod
    def observation_to_state(self, observation):
        pass

    @abstractmethod
    def state_values(self, state):
        pass

    @abstractmethod
    def action_value(self, state, action):
        pass

    def best_action(self, state):
        state_values = self.state_values(state)
        max_val = np.max(state_values)
        return np.random.choice([i for i in range(self.num_actions) if state_values[i] == max_val])

    def policy_action(self, observation):
        state = self.observation_to_state(observation)

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.best_action(state)

        self.prev_action = action
        self.prev_state = state
        return self.prev_action

    @abstractmethod
    def update(self, state, reward, done):
        pass

    @abstractmethod
    def plan_step(self):
        pass

    @abstractmethod
    def plan_episode(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class SimpleTD(Greedy, ABC):
    def __init__(self, epsilon, num_actions, gamma, step_size, state_size, start_values,
                 step_planning, episode_planning):
        super().__init__(epsilon, num_actions)
        self.step_size = step_size
        self.gamma = gamma
        self.state_size = state_size
        self.step_planning = step_planning
        self.episode_planning = episode_planning

        self.history = list()
        self._state_values = defaultdict(lambda: start_values + np.zeros(self.num_actions))

    def state_values(self, state):
        return self._state_values[state]

    def action_value(self, state, action):
        return self._state_values[state][action]

    def update(self, observation, reward, done):
        state = self.observation_to_state(observation)
        self.history.append((self.prev_state, self.prev_action, reward, state, done))

        self.weight_update(self.prev_state, self.prev_action, state, reward, done)

    @abstractmethod
    def target_val(self, reward, state, done):
        pass

    def weight_update(self, prev_state, prev_action, state, reward, done):
        current_val = self._state_values[prev_state][prev_action]
        self._state_values[prev_state][prev_action] += self.step_size * (
                self.target_val(reward, state, done) - current_val)

    def plan_step(self):
        if len(self.history) > self.step_planning:
            for _ in range(self.step_planning):
                prev_state, prev_action, reward, state, done = choice(self.history)
                self.weight_update(prev_state, prev_action, state, reward, done)

    def plan_episode(self):
        if len(self.history) > self.episode_planning:
            for _ in range(self.episode_planning):
                prev_state, prev_action, reward, state, done = choice(self.history)
                self.weight_update(prev_state, prev_action, state, reward, done)

    def save(self, path):
        dct = dict(self._state_values)
        with open(path, 'w') as f:
            json.dump(dct, f)

    def load(self, path):
        with open(path, 'r') as f:
            dct = json.load(f)
        self._state_values.update(dct)
