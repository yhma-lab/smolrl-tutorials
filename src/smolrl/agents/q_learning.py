from __future__ import annotations

import numpy as np


class Learner:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        # TODO: Implement the Q-learning update rule
        # q_update =
        # return q_update

        raise NotImplementedError("Q-learning update rule not implemented yet.")

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.rng = np.random.default_rng()

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # Find the indices where the Q-value equals the maximum value
            # Choose a random action from the indices where the Q-value is maximum
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = self.rng.choice(max_ids)
        return action


class QLearningAgent:
    def __init__(self, learning_rate, gamma, epsilon, state_size, action_size):
        self._learner = Learner(learning_rate, gamma, state_size, action_size)
        self._explorer = EpsilonGreedy(epsilon)

    def update(self, state, action, reward, new_state):
        q_update = self._learner.update(state, action, reward, new_state)
        self._learner.qtable[state, action] = q_update

    def choose_action(self, action_space, state):
        return self._explorer.choose_action(action_space, state, self._learner.qtable)

    def reset_learner(self):
        self._learner.reset_qtable()

    def get_qtable(self):
        return self._learner.qtable
