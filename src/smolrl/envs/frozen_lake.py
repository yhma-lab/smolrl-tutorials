from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pygame
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map
from pygame.event import Event
from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_UP, KEYDOWN

from ._utils import check_quit_event

__all__ = [
    "wait_human_input",
    "generate_random_map",
    "FROZEN_LAKE_V1",
    "ACTION_LABELS",
    "FrozenLakeParams",
    "FrozenLakeEnv",
]

FROZEN_LAKE_V1 = "FrozenLake-v1"
ACTION_LABELS = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}


@dataclass(kw_only=True)
class FrozenLakeParams:
    map_size: int
    """Number of tiles of one side of the squared environment"""
    is_slippery: bool
    """
    If true the player will move in intended direction with probability of
    1/3 else will move in either perpendicular direction with equal
    probability of 1/3 in both directions
    """
    proba_frozen: float
    """Probability that a tile is frozen"""
    render_mode: Literal["human", "rgb_array"]
    "Render mode"
    seed: int | None
    """Random seed"""


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
