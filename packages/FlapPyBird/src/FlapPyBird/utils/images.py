import random
from typing import List, Tuple

import pygame

from .constants import (
    ASSETS,
    BACKGROUNDS_SETUP,
    PIPES_SETUP,
    PLAYERS_SETUP,
    BackgroundColor,
    PipeColor,
    PlayerColor,
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
        player: PlayerColor | None = None,
        bg: BackgroundColor | None = None,
        pipe: PipeColor | None = None,
    ) -> None:
        self.numbers = list(
            (
                pygame.image.load(ASSETS / "sprites" / f"{num}.png").convert_alpha()
                for num in range(10)
            )
        )

        # game over sprite
        self.game_over = pygame.image.load(
            ASSETS / "sprites" / "gameover.png"
        ).convert_alpha()
        # welcome_message sprite for welcome screen
        self.welcome_message = pygame.image.load(
            ASSETS / "sprites" / "message.png"
        ).convert_alpha()
        # base (ground) sprite
        self.base = pygame.image.load(ASSETS / "sprites" / "base.png").convert_alpha()

        self.setup_backgroud(bg)
        self.setup_player(player)
        self.setup_pipe(pipe)

    def setup_backgroud(self, bg: BackgroundColor | None) -> None:
        if bg is None:
            bg: BackgroundColor = random.choice(list(BACKGROUNDS_SETUP.keys()))  # type: ignore[reportAssignmentType]

        self.background = pygame.image.load(BACKGROUNDS_SETUP[bg]).convert()

    def setup_player(self, player: PlayerColor | None) -> None:
        if player is None:
            player: PlayerColor = random.choice(tuple(PLAYERS_SETUP.keys()))  # type: ignore[reportAssignmentType]

        self.player = (
            pygame.image.load(PLAYERS_SETUP[player][0]).convert_alpha(),
            pygame.image.load(PLAYERS_SETUP[player][1]).convert_alpha(),
            pygame.image.load(PLAYERS_SETUP[player][2]).convert_alpha(),
        )

    def setup_pipe(self, pipe: PipeColor | None) -> None:
        if pipe is None:
            pipe: PipeColor = random.choice(tuple(PIPES_SETUP.keys()))  # type: ignore[reportAssignmentType]

        self.pipe = (
            pygame.transform.flip(
                pygame.image.load(PIPES_SETUP[pipe]).convert_alpha(), False, True
            ),
            pygame.image.load(PIPES_SETUP[pipe]).convert_alpha(),
        )
