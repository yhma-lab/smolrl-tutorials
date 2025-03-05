from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from FlapPyBird.constants import FPS
from FlapPyBird.flappy import Flappy
from FlapPyBird.utils.constants import BackgroundColor, PipeColor, PlayerColor
from gymnasium import Env, spaces

from .common import UP, ActType, ObsType, RenderMode


@dataclass(kw_only=True)
class FlappyBirdSimpleParams:
    pipe_gap: int = 100
    player: PlayerColor | None = "yellow"
    pipe: PipeColor | None = "green"
    background: BackgroundColor | None = "day"
    screen_size: tuple[int, int] = (288, 512)
    normalize_obs: bool = True
    render_mode: RenderMode = "rgb_array"


class FlappyBirdSimpleEnv(Env[ObsType, ActType]):
    """A customizable Flappy Bird environment (that yields coords as observations) with gymnasium-like interface."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        pipe_gap: int = 100,
        player: PlayerColor | None = "yellow",
        pipe: PipeColor | None = "green",
        background: BackgroundColor | None = "day",
        screen_size: tuple[int, int] = (288, 512),
        normalize_obs: bool = True,
        render_mode: RenderMode = "human",
    ):
        """
        Initialize the Flappy Bird environment.

        Args:
            size (int): Size of the grid (size x size)
            random_seed (int, optional): Seed for reproducibility
        """
        self._screen_size = screen_size
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap
        _, h = self._screen_size

        self._game = Flappy(
            player=player,
            pipe=pipe,
            bg=background,
            screen_size=screen_size,
            silent=True,
        )

        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)  # type: ignore
        self.observation_space = spaces.Box(-h, h, shape=(2,), dtype=np.float64)

    def _get_observation(self):
        pipes = self._game.pipes
        player = self._game.player

        iterator = zip(pipes.upper, pipes.lower)
        while ul := next(iterator, None):
            upper_pipe, lower_pipe = ul
            dx = lower_pipe.x + lower_pipe.w / 2 - (player.x - player.w / 2)
            dx += 3  # extra distance to compensate for the buggy hit-box
            if dx >= 0:
                break
        else:
            raise RuntimeError("No pipes found, which is not by design, game is broken")

        upper_bottom = upper_pipe.y + upper_pipe.h
        lower_top = lower_pipe.y

        dy = (upper_bottom + lower_top) / 2 - (player.cy)

        if self._normalize_obs:
            dx /= self._screen_size[0]
            dy /= self._screen_size[1]

        return np.array([dx, dy])

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        prev_score = self._game.score.score

        if action == UP:
            self._game.player.flap()

        obs = self._get_observation()

        self._game._tick_play()

        alive = self._game.is_player_collided()

        new_score = self._game.score.score

        if new_score > prev_score:
            reward = 10
        elif alive:
            reward = 1
        else:
            reward = -50

        terminated = not alive
        info = {"score": new_score}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Resets the environment (starts a new game)."""
        self._game.reset()
        return self._get_observation(), {"score": self._game.score}

    def render(self) -> None:
        """Renders the next frame."""
        pygame.display.update()

    def close(self):
        """Closes the environment."""
        self._game.close()


FLAPPY_BIRD_SIMPLE_V1 = "FlappyBird-simple-v1"


def register_flappy_bird():
    """Register the Flappy Bird environment with gymnasium."""
    gym.register(
        FLAPPY_BIRD_SIMPLE_V1,
        entry_point=lambda **kwargs: FlappyBirdSimpleEnv(**kwargs),
    )


register_flappy_bird()
