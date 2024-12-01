import importlib.resources
from pathlib import Path
from typing import Literal

ASSETS = Path(str(importlib.resources.files("FlapPyBird") / "assets"))


# list of all possible players (tuple of 3 positions of flap)
Players = Literal["red", "blue", "yellow"]
PLAYERS_SETUP = {
    # red bird
    "red": (
        ASSETS / "sprites" / "redbird-upflap.png",
        ASSETS / "sprites" / "redbird-midflap.png",
        ASSETS / "sprites" / "redbird-downflap.png",
    ),
    # blue bird
    "blue": (
        ASSETS / "sprites" / "bluebird-upflap.png",
        ASSETS / "sprites" / "bluebird-midflap.png",
        ASSETS / "sprites" / "bluebird-downflap.png",
    ),
    # yellow bird
    "yellow": (
        ASSETS / "sprites" / "yellowbird-upflap.png",
        ASSETS / "sprites" / "yellowbird-midflap.png",
        ASSETS / "sprites" / "yellowbird-downflap.png",
    ),
}

# list of backgrounds
Backgrounds = Literal["day", "night", "black"]
BACKGROUNDS_SETUP = {
    "day": ASSETS / "sprites" / "background-day.png",
    "night": ASSETS / "sprites" / "background-night.png",
    "black": ASSETS / "sprites" / "background-black.png",
}

# list of pipes
Pipes = Literal["green", "red"]
PIPES_SETUP = {
    "green": ASSETS / "sprites" / "pipe-green.png",
    "red": ASSETS / "sprites" / "pipe-red.png",
}
