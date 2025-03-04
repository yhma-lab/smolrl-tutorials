from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pygame
from pygame.event import Event
from pygame.locals import FINGERDOWN, K_SPACE, K_UP, KEYDOWN, MOUSEBUTTONDOWN

from smolrl.envs._utils import check_quit_event

UP = 1
NOOP = 0
ActType = Literal[1, 0]
ObsType = npt.NDArray[np.float64]
RenderMode = Literal["human", "rgb_array"]

ACTION_LABELS: dict[str, ActType] = {"UP": UP, "NOOP": NOOP}


def get_action_from(event: Event) -> ActType:
    if event.type == FINGERDOWN:
        return UP

    if event.type != KEYDOWN or event.type != MOUSEBUTTONDOWN:
        return NOOP

    m_left, _, _ = pygame.mouse.get_pressed()
    if m_left or event.key == K_SPACE or event.key == K_UP:
        return UP
    else:
        return NOOP


def wait_human_input() -> int:
    while True:
        for event in pygame.event.get():
            check_quit_event(event)
            action = get_action_from(event)
            if action is not None:
                return action
