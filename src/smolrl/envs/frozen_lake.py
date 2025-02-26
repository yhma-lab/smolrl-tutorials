from __future__ import annotations

import sys
from typing import NoReturn

import pygame
from pygame.event import Event
from pygame.locals import K_DOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_UP, KEYDOWN, QUIT


def get_action_from(event: Event) -> int | None:
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
        raise RuntimeError("Unreachable code")


def check_quit_event(event: Event) -> NoReturn | None:
    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        pygame.quit()
        sys.exit()


def wait_player_input() -> int:
    while True:
        for event in pygame.event.get():
            check_quit_event(event)
            action = get_action_from(event)
            if action is not None:
                return action


def wait_quit() -> NoReturn | None:
    for event in pygame.event.get():
        check_quit_event(event)
