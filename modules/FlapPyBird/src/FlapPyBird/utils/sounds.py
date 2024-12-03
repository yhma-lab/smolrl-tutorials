import sys

import pygame

from .constants import ASSETS


class Sounds:
    die: pygame.mixer.Sound
    hit: pygame.mixer.Sound
    point: pygame.mixer.Sound
    swoosh: pygame.mixer.Sound
    wing: pygame.mixer.Sound

    def __init__(self) -> None:
        if "win" in sys.platform:
            ext = "wav"
        else:
            ext = "ogg"

        self.die = pygame.mixer.Sound(ASSETS / "audio" / f"die.{ext}")
        self.hit = pygame.mixer.Sound(ASSETS / "audio" / f"hit.{ext}")
        self.point = pygame.mixer.Sound(ASSETS / "audio" / f"point.{ext}")
        self.swoosh = pygame.mixer.Sound(ASSETS / "audio" / f"swoosh.{ext}")
        self.wing = pygame.mixer.Sound(ASSETS / "audio" / f"wing.{ext}")


class FakeSound:
    def __init__(self) -> None:
        pass

    def play(self) -> None:
        """Do nothing."""

    def stop(self) -> None:
        """Do nothing."""

    def fadeout(self, ms: int) -> None:
        """Do nothing."""


class SilentSounds:
    die: FakeSound
    hit: FakeSound
    point: FakeSound
    swoosh: FakeSound
    wing: FakeSound

    def __init__(self) -> None:
        self.die = FakeSound()
        self.hit = FakeSound()
        self.point = FakeSound()
        self.swoosh = FakeSound()
        self.wing = FakeSound()
