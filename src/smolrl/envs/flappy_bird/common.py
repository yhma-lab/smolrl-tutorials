from __future__ import annotations

from typing import Literal, NoReturn

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


def _get_action_from(event: Event) -> ActType:
    if event.type == FINGERDOWN:
        return UP
    elif event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
        return UP
    elif event.type == MOUSEBUTTONDOWN:
        m_left, _, _ = pygame.mouse.get_pressed()
        return UP if m_left else NOOP
    else:
        return NOOP


def wait_human_input() -> ActType:
    action = NOOP
    for event in pygame.event.get():
        check_quit_event(event)
        action = _get_action_from(event)
        break
    return action
