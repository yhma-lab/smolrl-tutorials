from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pygame
from FlapPyBird.flappy import Flappy
from gymnasium import Env, spaces

from .game_logic import FlappyBirdLogic
from .renderer import FlappyBirdRenderer

ObsType = npt.NDArray[np.float64]


UP = 1
NOOP = 0
ActType = Literal[UP, NOOP]


class FlappyBirdSimpleEnv(Env[ObsType, ActType]):
    """A customizable Flappy Bird environment with gymnasium-like interface."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        screen_size: tuple[int, int] = (288, 512),
        normalize_obs: bool = True,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        background: str | None = "day",
        render_mode: str = "human",
    ):
        """
        Initialize the Flappy Bird environment.

        Args:
            size (int): Size of the grid (size x size)
            random_seed (int, optional): Seed for reproducibility
        """

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(2,), dtype=np.float32
        )

        self._screen_size = screen_size
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap

        self._game = None
        self._renderer = None

        self._bird_color = bird_color
        self._pipe_color = pipe_color
        self._bg_type = background

    def _get_observation(self):
        up_pipe = low_pipe = None
        h_dist = 0
        for up_pipe, low_pipe in zip(self._game.upper_pipes, self._game.lower_pipes):
            h_dist = (
                low_pipe["x"]
                + PIPE_WIDTH / 2
                - (self._game.player_x - PLAYER_WIDTH / 2)
            )
            h_dist += 3  # extra distance to compensate for the buggy hit-box
            if h_dist >= 0:
                break

        upper_pipe_y = up_pipe["y"] + PIPE_HEIGHT
        lower_pipe_y = low_pipe["y"]
        player_y = self._game.player_y

        v_dist = (upper_pipe_y + lower_pipe_y) / 2 - (player_y + PLAYER_HEIGHT / 2)

        if self._normalize_obs:
            h_dist /= self._screen_size[0]
            v_dist /= self._screen_size[1]

        return np.array([h_dist, v_dist])

    def step(self, action):
        alive = self._game.update_state(action)
        obs = self._get_observation()

        reward = 1

        terminated = not alive
        info = {"score": self._game.score}

        return obs, reward, terminated, False, info

    def reset(self):
        """Resets the environment (starts a new game)."""
        self._game = FlappyBirdLogic(
            screen_size=self._screen_size, pipe_gap_size=self._pipe_gap
        )
        if self._renderer is not None:
            self._renderer.game = self._game

        return self._get_observation()

    def render(self, mode="human") -> None:
        """Renders the next frame."""
        if self._renderer is None:
            self._renderer = FlappyBirdRenderer(
                screen_size=self._screen_size,
                bird_color=self._bird_color,
                pipe_color=self._pipe_color,
                background=self._bg_type,
            )
            self._renderer.game = self._game
            self._renderer.make_display()

        self._renderer.draw_surface(show_score=True)
        self._renderer.update_display()

    def close(self):
        """Closes the environment."""
        if self._renderer is not None:
            pygame.display.quit()
            self._renderer = None


FLAPPY_BIRD_SIMPLE_V1 = "FlappyBird-simple-v1"


def register_flappy_bird():
    """Register the Flappy Bird environment with gymnasium."""
    gym.register(FLAPPY_BIRD_SIMPLE_V1, entry_point=FlappyBirdSimpleEnv)


register_flappy_bird()
