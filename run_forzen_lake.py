# pyright: reportPossiblyUnboundVariable=false, reportAttributeAccessIssue=false
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
    generate_random_map,
    wait_human_input,
)
from smolrl.vis import plot_q_values_map, plot_states_actions_distribution

console = Console()


@dataclass
class Params:
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    render_mode: RenderEnum  # Render mode
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved


def init_env(params: Params):
    env = gym.make(
        FROZEN_LAKE_V1,
        is_slippery=params.is_slippery,
        render_mode=params.render_mode,
        desc=generate_random_map(size=params.map_size, p=params.proba_frozen),
    )

    action_size = env.action_space.n
    state_size = env.observation_space.n
    console.print("Environment initialized ...")
    console.print(f"Action size: {action_size}")
    console.print(f"State size: {state_size}")
    console.print(f"Map size: {params.map_size}x{params.map_size}")
    return env


def run_experiments(env: gym.Env, params: Params, run_mode: PlayEnum = PlayEnum.human):
    action_size = env.action_space.n
    state_size = env.observation_space.n
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, state_size, action_size))
    all_states = []
    all_actions = []

    agent = QLearningAgent(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        epsilon=params.epsilon,
        state_size=state_size,
        action_size=action_size,
    )
    console.print("Agent initialized ...")

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        agent.reset_learner()

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
                action = agent.choose_action(action_space=env.action_space, state=state)

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, new_state)

                done = terminated or truncated
                total_rewards += reward  # pyright: ignore[reportOperatorIssue]
                step += 1

                if done:
                    if params.render_mode == "human":
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

            # console.print(
            #     f"Finish {run}/{params.n_runs} - Episode {episode}/{params.total_episodes}"
            # )

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step

            # console.print(
            #     f"Episode: {episode} - Total reward: {total_rewards} - Steps: {step}"
            # )

        qtables[run, :, :] = agent.get_qtable()
        # console.print("Q-table", agent.get_qtable())

    return rewards, steps, episodes, qtables, all_states, all_actions


def main(
    play_mode: PlayEnum = typer.Option(
        PlayEnum.human, show_choices=True, help="Run mode: `human` or `agent`"
    ),
    render_mode: RenderEnum = typer.Option(  # type: ignore[assignment]
        RenderEnum.human, help="Render mode: `human`, `rgb_array`"
    ),
    expname: str | None = None,
):
    exp_dirname = expname or int(time.monotonic())
    params = Params(
        total_episodes=2000,
        learning_rate=0.8,
        gamma=0.95,
        epsilon=0.1,
        map_size=5,
        is_slippery=False,
        render_mode=render_mode,
        n_runs=20,
        proba_frozen=0.9,
        savefig_folder=Path(f"./run/{exp_dirname}"),
    )
    env = init_env(params)

    if play_mode == PlayEnum.human:
        if render_mode != RenderEnum.human:
            raise ValueError(
                "Environment must be in human render mode when `play_mode` == 'human'"
            )
        human_play(env, wait_human_input)
        return

    params.savefig_folder.mkdir(parents=True, exist_ok=True)
    rewards, steps, episodes, qtables, all_states, all_actions = run_experiments(params)

    # Save the results in dataframes and plot them
    # res, st = postprocess(episodes, params, rewards, steps, params.map_size)
    qtable = qtables.mean(axis=0)  # Average the Q-table between runs

    plot_states_actions_distribution(
        states=all_states,
        actions=all_actions,
        map_size=params.map_size,
        labels=ACTION_LABELS,
        savefig_folder=params.savefig_folder,
        show=False,
    )
    plot_q_values_map(
        qtable=qtable,
        map_size=params.map_size,
        savefig_folder=params.savefig_folder,
        show=False,
    )
    plt.show()

    # TODO: How to compare the steps and rewards in different map sizes?
    # plot_steps_and_rewards(res_all, st_all)


if __name__ == "__main__":
    typer.run(main)
