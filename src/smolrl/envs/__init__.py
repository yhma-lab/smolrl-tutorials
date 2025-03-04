from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

from ._utils import human_play
from .flappy_bird import FLAPPY_BIRD_SIMPLE_V1, FlappyBirdSimpleEnv
from .frozen_lake import FROZEN_LAKE_V1

__all__ = [
    "FLAPPY_BIRD_SIMPLE_V1",
    "FlappyBirdSimpleEnv",
    "FROZEN_LAKE_V1",
    "FrozenLakeEnv",
    "RenderEnum",
    "PlayEnum",
    "human_play",
]

RENDER_MODE = Literal["human", "rgb_array"]
PLAY_MODE = Literal["human", "agent"]


class RenderEnum(StrEnum):
    human = "human"
    rgb_array = "rgb_array"
    # ansi = "ansi"


class PlayEnum(StrEnum):
    human = "human"
    agent = "agent"


@dataclass(frozen=True)
class RunMode:
    player: PlayEnum = PlayEnum.human
    render: RenderEnum = RenderEnum.human


AVAILABLE_MODES = {
    RunMode(PlayEnum.human, RenderEnum.human),
    RunMode(PlayEnum.agent, RenderEnum.human),
    RunMode(PlayEnum.agent, RenderEnum.rgb_array),
}
