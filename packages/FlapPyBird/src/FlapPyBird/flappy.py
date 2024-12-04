# import asyncio
import sys
from contextlib import contextmanager
from typing import TypedDict

import pygame
from numpy.typing import NDArray
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .constants import FPS
from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, SilentSounds, Sounds, Window
from .utils.constants import Backgrounds, Players
from .utils.constants import Pipes as PipesT


class GameState(TypedDict):
    env: NDArray
    action: int
    reward: float
    terminal: bool


class Flappy:
    def __init__(
        self,
        silent: bool = False,
        player: Players | None = None,
        bg: Backgrounds | None = None,
        pipe: PipesT | None = None,
    ):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")

        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images(player=player, bg=bg, pipe=pipe)

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=FPS,
            window=window,
            images=images,
            sounds=Sounds() if not silent else SilentSounds(),
        )

    def init_entities(self):
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.welcome_message = WelcomeMessage(self.config)
        self.game_over_message = GameOver(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)

    def start(self):
        while True:
            self.init_entities()
            self.splash()
            self.play()
            self.game_over()

    def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            # await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
            event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
            pygame.display.update()
            # await asyncio.sleep(0)
            self.config.tick()

    def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            # await asyncio.sleep(0)
