import random
from typing import List, Tuple

import pygame

from .constants import (
    BACKGROUNDS_SETUP,
    PIPES_SETUP,
    PLAYERS_SETUP,
    Backgrounds,
    Pipes,
    Players,
)


class Images:
    numbers: List[pygame.Surface]
    game_over: pygame.Surface
    welcome_message: pygame.Surface
    base: pygame.Surface
    background: pygame.Surface
    player: Tuple[pygame.Surface, ...]
    pipe: Tuple[pygame.Surface, ...]

    def __init__(
        self,
        *,
        player: Players | None = None,
        bg: Backgrounds | None = None,
        pipe: Pipes | None = None,
    ) -> None:
        self.numbers = list(
            (
                pygame.image.load(f"assets/sprites/{num}.png").convert_alpha()
                for num in range(10)
            )
        )

        # game over sprite
        self.game_over = pygame.image.load(
            "assets/sprites/gameover.png"
        ).convert_alpha()
        # welcome_message sprite for welcome screen
        self.welcome_message = pygame.image.load(
            "assets/sprites/message.png"
        ).convert_alpha()
        # base (ground) sprite
        self.base = pygame.image.load("assets/sprites/base.png").convert_alpha()

        self.setup_backgroud(bg)
        self.setup_player(player)
        self.setup_pipe(pipe)

    def setup_backgroud(self, bg: Backgrounds | None) -> None:
        if bg is None:
            bg: Backgrounds = random.choice(list(BACKGROUNDS_SETUP.keys()))  # type: ignore[reportAssignmentType]

        self.background = pygame.image.load(BACKGROUNDS_SETUP[bg]).convert()

    def setup_player(self, player: Players | None) -> None:
        if player is None:
            player: Players = random.choice(tuple(PLAYERS_SETUP.keys()))  # type: ignore[reportAssignmentType]

        self.player = (
            pygame.image.load(PLAYERS_SETUP[player][0]).convert_alpha(),
            pygame.image.load(PLAYERS_SETUP[player][1]).convert_alpha(),
            pygame.image.load(PLAYERS_SETUP[player][2]).convert_alpha(),
        )

    def setup_pipe(self, pipe: Pipes | None) -> None:
        if pipe is None:
            pipe: Pipes = random.choice(tuple(PIPES_SETUP.keys()))  # type: ignore[reportAssignmentType]

        self.pipe = (
            pygame.transform.flip(
                pygame.image.load(PIPES_SETUP[pipe]).convert_alpha(), False, True
            ),
            pygame.image.load(PIPES_SETUP[pipe]).convert_alpha(),
        )
