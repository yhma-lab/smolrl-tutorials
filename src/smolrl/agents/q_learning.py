from __future__ import annotations

from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, Discrete, Space
from numpy import random

StateSize = Literal[0] | int
ActionSize = Literal[0] | int

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class QLearningParams(Generic[ObsType, ActType]):
    state_space: Space[ObsType]
    action_space: Space[ActType]
    learning_rate: float
    gamma: float
    epsilon: float


@dataclass(kw_only=True)
class BaseQFunc(Generic[ObsType, ActType]):
    state_space: Space[ObsType]
    action_space: Space[ActType]

    @abstractmethod
    def get_q_value(self, state: ObsType, action: ActType):
        raise NotImplementedError

    @abstractmethod
    def max_q_prime(self, state: ObsType):
        raise NotImplementedError

    @abstractmethod
    def update(self, q_update, state: ObsType, action: ActType):
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the Q-table."""
        raise NotImplementedError


@dataclass(kw_only=True)
class DiscreteQFunc(BaseQFunc):
    _qtable: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self):
        state_szie = self.state_space.n  # pyright: ignore[reportAttributeAccessIssue]
        action_size = self.action_space.n  # pyright: ignore[reportAttributeAccessIssue]
        self._qtable = np.zeros((state_szie, action_size))

    def max_q_prime(self, state):
        return np.max(self._qtable[state, :])

    def get_q_value(self, state, action):
        return self._qtable[state, action]

    def update(self, q_update, state, action):
        self._qtable[state, action] = q_update

    def reset(self) -> None:
        self._qtable.fill(0)


@dataclass(kw_only=True)
class ContinuousQFunc(BaseQFunc):
    memory_size: int = 10000
    _qfunc: deque = field(init=False)

    def __post_init__(self):
        self._qfunc = deque(maxlen=self.memory_size)

    def reset(self) -> None:
        self._qfunc.clear()


@dataclass(kw_only=True)
class Learner(Generic[ObsType, ActType]):
    learning_rate: float
    gamma: float
    state_space: Space[ObsType]
    action_space: Space[ActType]
    qfunc: BaseQFunc = field(init=False)

    def __post_init__(self):
        if isinstance(self.state_space, Box):
            QFuncCls = ContinuousQFunc
        elif isinstance(self.state_space, Discrete):
            QFuncCls = DiscreteQFunc
        else:
            raise NotImplementedError(
                f"Q Learner for state space {self.state_space} doesn't implement now."
            )

        self.qfunc = QFuncCls(
            action_space=self.action_space, state_space=self.state_space
        )

    def calc_q_update(
        self, state: ObsType, action: ActType, reward: Any, new_state: ObsType
    ):
        delta = (
            reward
            + self.gamma * self.qfunc.max_q_prime(new_state)
            - self.qfunc.get_q_value(state, action)
        )
        q_update = self.qfunc.get_q_value(state, action) + self.learning_rate * delta
        return q_update

    def update(self, q_update, state, action):
        self.qfunc.update(q_update, state, action)

    def reset(self):
        """Reset learner (Q-func/Q-table), clear all history memory."""
        self.qfunc.reset()


@dataclass(kw_only=True)
class EpsilonGreedy(Generic[ObsType, ActType]):
    epsilon: float
    learner: Learner[ObsType, ActType]
    state_space: Space[ObsType]
    action_space: Space[ActType]
    rng: random.Generator = field(default_factory=lambda: random.default_rng())

    def choose_action(self, state: ObsType) -> ActType:
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = self.action_space.sample()
        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # Find the indices where the Q-value equals the maximum value
            # Choose a random action from the indices where the Q-value is maximum
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = self.rng.choice(max_ids)
        return action


class QLearningAgent(Generic[ObsType, ActType]):
    def __init__(
        self,
        *,
        state_space: Space[ObsType],
        action_space: Space[ActType],
        learning_rate: float,
        gamma: float,
        epsilon: float,
    ):
        self._learner: Learner[ObsType, ActType] = Learner(
            learning_rate=learning_rate,
            gamma=gamma,
            state_space=state_space,
            action_space=action_space,
        )
        self._explorer = EpsilonGreedy(
            epsilon=epsilon,
            learner=self._learner,
            state_space=state_space,
            action_space=action_space,
        )

    def update(self, state: ObsType, action: ActType, reward: Any, new_state: ObsType):
        """Update the Q-func."""
        q_update = self._learner.calc_q_update(state, action, reward, new_state)
        self._learner.update(q_update, state, action)

    def choose_action(self, state: ObsType) -> ActType:
        return self._explorer.choose_action(state)

    def reset(self):
        self._learner.reset()
