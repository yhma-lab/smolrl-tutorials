# pyright: reportAttributeAccessIssue=false
# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import typer
from matplotlib import pyplot as plt
from rich.console import Console
from tqdm import tqdm

from smolrl.agents import QLearningAgent
from smolrl.envs import PlayEnum, RenderEnum, human_play
from smolrl.envs.frozen_lake import (
    ACTION_LABELS,
    FROZEN_LAKE_V1,
    FrozenLakeParams,
    generate_random_map,
    wait_human_input,
)
from smolrl.vis import plot_q_table_map, plot_steps_and_rewards

console = Console()


@dataclass
class TrainParams:
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    n_runs: int  # Number of runs
    savefig_folder: Path  # Root folder where plots are saved


def init_env(params: FrozenLakeParams):
    env = gym.make(
        FROZEN_LAKE_V1,
        is_slippery=params.is_slippery,
        render_mode=params.render_mode,
        desc=generate_random_map(
            size=params.map_size, p=params.proba_frozen, seed=params.seed
        ),
    )

    action_size = env.action_space.n
    state_size = env.observation_space.n
    console.print("Environment initialized ...")
    console.print(f"Action size: {action_size}")
    console.print(f"State size: {state_size}")
    console.print(f"Map size: {params.map_size}x{params.map_size}")
    return env


def run_experiments(env: gym.Env, params: TrainParams, vis: bool = False):
    action_size = env.action_space.n
    state_size = env.observation_space.n
    render_mode = env.render_mode

    rewards = np.zeros((params.n_runs, params.total_episodes))
    steps = np.zeros((params.n_runs, params.total_episodes))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, state_size, action_size))

    all_states = []
    all_actions = []

    agent = QLearningAgent(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        epsilon=params.epsilon,
        obs_space=env.observation_space,
        action_space=env.action_space,
    )
    console.print("Agent initialized ...")

    # if vis:
    #     fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        console.print(f"Start to run {run + 1}/{params.n_runs}...")
        agent.reset()

        tic = time.monotonic()
        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            # init
            state = env.reset()[0]
            step = 0
            done = False
            total_rewards = 0.0

            # training
            while not done:
                action = agent.choose_action(state=state)

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, new_state)

                done = terminated or truncated
                total_rewards += reward  # type: ignore
                step += 1

                if done:
                    if render_mode == "human":
                        if reward > 0:  # pyright: ignore[reportOperatorIssue]
                            console.print(
                                "Agent reached the goal! Restarting ...", style="green"
                            )
                        else:
                            console.print("Agent died! Restarting ...", style="red")
                    # time.sleep(0.1)
                    state = env.reset()[0]
                else:
                    state = new_state

            rewards[run, episode] = total_rewards
            steps[run, episode] = step

        toc = time.monotonic()
        qtables[run, :, :] = agent.learner.get_q_func()
        console.print(
            f"Finish to run {run + 1}/{params.n_runs}\n"
            f"  total {params.total_episodes} episodes\n"
            f"  total steps: {steps[:, run].sum()}\n"
            f"  total rewards: {rewards[:, run].sum()}\n"
            f"  time: {toc - tic:.2f} seconds"
        )

    lastframe = env.render()
    return rewards, steps, episodes, qtables, all_states, all_actions, lastframe


def main(
    play_mode: PlayEnum = typer.Option(
        PlayEnum.human, show_choices=True, help="Run mode: `human` or `agent`"
    ),
    render_mode: RenderEnum = typer.Option(  # type: ignore[assignment]
        RenderEnum.human, help="Render mode: `human`, `rgb_array`"
    ),
    vis: bool = typer.Option(True, help="Visualize the training process"),
    expname: str | None = None,
):
    exp_dirname = expname or int(time.monotonic())
    env_params = FrozenLakeParams(
        map_size=5,
        is_slippery=False,
        proba_frozen=0.85,
        render_mode=render_mode.value,
        seed=123,
    )
    train_params = TrainParams(
        total_episodes=2000,
        learning_rate=0.8,
        gamma=0.95,
        epsilon=0.1,
        n_runs=10,
        savefig_folder=Path(f"./run/{exp_dirname}"),
    )
    env = init_env(env_params)

    if play_mode == PlayEnum.human:
        if render_mode != RenderEnum.human:
            raise ValueError(
                "Environment must be in human render mode when `play_mode` == 'human'"
            )
        human_play(env, wait_human_input)
        env.close()
        return

    # fmt: off
    rewards, steps, episodes, qtables, all_states, all_actions, last_frame = run_experiments(
        env=env,
        params=train_params,
        vis=vis,
    )
    # fmt: on
    env.close()

    qtable = qtables.mean(axis=0)  # Average the Q-table between runs

    # train_params.savefig_folder.mkdir(parents=True, exist_ok=True)
    plot_steps_and_rewards(
        rewards=rewards,
        steps=steps,
        # savefig_folder=train_params.savefig_folder,
        show=False,
    )
    plot_q_table_map(
        last_frame=last_frame,
        qtable=qtable,
        map_size=env_params.map_size,
        # savefig_folder=train_params.savefig_folder,
        show=False,
    )
    plt.show()

    # TODO: How to compare the steps and rewards in different map sizes?
    # plot_steps_and_rewards(res_all, st_all)


if __name__ == "__main__":
    typer.run(main)
