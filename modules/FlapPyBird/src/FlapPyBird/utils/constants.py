from typing import Literal

# list of all possible players (tuple of 3 positions of flap)
Players = Literal["red", "blue", "yellow"]
PLAYERS_SETUP = {
    # red bird
    "red": (
        "assets/sprites/redbird-upflap.png",
        "assets/sprites/redbird-midflap.png",
        "assets/sprites/redbird-downflap.png",
    ),
    # blue bird
    "blue": (
        "assets/sprites/bluebird-upflap.png",
        "assets/sprites/bluebird-midflap.png",
        "assets/sprites/bluebird-downflap.png",
    ),
    # yellow bird
    "yellow": (
        "assets/sprites/yellowbird-upflap.png",
        "assets/sprites/yellowbird-midflap.png",
        "assets/sprites/yellowbird-downflap.png",
    ),
}

# list of backgrounds
Backgrounds = Literal["day", "night", "black"]
BACKGROUNDS_SETUP = {
    "day": "assets/sprites/background-day.png",
    "night": "assets/sprites/background-night.png",
    "black": "assets/sprites/background-black.png",
}

# list of pipes
Pipes = Literal["green", "red"]
PIPES_SETUP = {
    "green": "assets/sprites/pipe-green.png",
    "red": "assets/sprites/pipe-red.png",
}
