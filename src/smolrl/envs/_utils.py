from __future__ import annotations

import sys
from collections.abc import Callable
from typing import NoReturn, TypeVar

import pygame
from gymnasium import Env
from pygame.event import Event
from pygame.locals import K_ESCAPE, KEYDOWN, QUIT


def check_quit_event(event: Event) -> NoReturn | None:
    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        pygame.quit()
        sys.exit()


def wait_quit() -> NoReturn | None:
    for event in pygame.event.get():
        check_quit_event(event)


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def human_play(
    env: Env[ObsType, ActType], wait_human_input: Callable[[], ActType]
) -> NoReturn | None:
    """Human play loop."""

    if env.render_mode != "human":
        raise ValueError("Environment must be in human render mode")

    env.reset()
    while True:
        action = wait_human_input()
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            env.reset()
