from __future__ import annotations

import pygame
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from pygame.event import Event
from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_UP, KEYDOWN

from smolrl.envs._utils import check_quit_event

__all__ = ["wait_human_input", "generate_random_map", "FROZEN_LAKE_V1", "ACTION_LABELS"]

FROZEN_LAKE_V1 = "FrozenLake-v1"
ACTION_LABELS = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}


def _get_action_from(event: Event) -> int | None:
    if event.type != KEYDOWN:
        return None

    if event.key == K_RIGHT:
        return 2
    elif event.key == K_LEFT:
        return 0
    elif event.key == K_DOWN:
        return 1
    elif event.key == K_UP:
        return 3
    else:
        return None


def wait_human_input() -> int:
    while True:
        for event in pygame.event.get():
            check_quit_event(event)
            action = _get_action_from(event)
            if action is not None:
                return action
