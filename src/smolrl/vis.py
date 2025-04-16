from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.figure import Figure


def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def calc_std_bounds(
    data: npt.NDArray[np.float64],
    nsigma: int = 3,
    clip: tuple[float, float] | None = (-np.inf, np.inf),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the mean and standard deviation boundary of the data."""
    mean = data.mean(axis=0)
    stdvar = data.std(axis=0, ddof=1)
    if clip is None:
        return mean, mean - nsigma * stdvar, mean + nsigma * stdvar
    else:
        lower = np.clip(mean - nsigma * stdvar, clip[0], clip[1])
        upper = np.clip(mean + nsigma * stdvar, clip[0], clip[1])
        return mean, lower, upper


def calc_minmax_bounds(
    data: npt.NDArray[np.float64], nsigma: int = 3
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the mean and standard deviation boundary of the data."""
    mean = data.mean(axis=0)
    lower = data.min(axis=0)
    upper = data.max(axis=0)
    return mean, lower, upper


def plot_q_table_map(
    last_frame, qtable, map_size, savefig_folder: Path | None = None, show: bool = True
) -> Figure:
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    axes[0].imshow(last_frame)
    axes[0].axis("off")
    axes[0].set_title("Last frame of Game")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=axes[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in axes[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    if savefig_folder:
        img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
        fig.savefig(savefig_folder / img_title, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig


def plot_steps_and_rewards(
    episodes: npt.NDArray[np.int64],
    rewards: npt.NDArray[np.float64],
    steps: npt.NDArray[np.float64],
    savefig_folder: Path | None = None,
    show: bool = True,
):
    """Plot the steps and rewards from dataframes."""
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))

    s_mean, s_lb, s_ub = calc_std_bounds(steps, clip=(0, np.inf))
    axes[0].plot(episodes, s_mean, label="Mean steps")
    axes[0].fill_between(
        episodes, s_lb, s_ub, alpha=0.6, label=r"confidence interval: 3$\sigma$"
    )
    axes[0].set(ylabel="Averaged steps number per Run")
    axes[0].legend()

    r_mean, r_lb, r_ub = calc_std_bounds(rewards.cumsum(axis=1), clip=(0, np.inf))
    axes[1].plot(episodes, r_mean, label="Cumulated rewards")
    axes[1].fill_between(
        episodes, r_lb, r_ub, alpha=0.6, label=r"confidence interval: 3$\sigma$"
    )
    axes[1].set(ylabel="Cumulated rewards per Run", xlabel="Episodes")
    axes[1].legend()

    fig.tight_layout()
    if savefig_folder:
        img_title = "frozenlake_steps_and_rewards.png"
        fig.savefig(savefig_folder / img_title, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig


def plot_steps_and_rewards_for_all_exps(
    map_sizes: Iterable[int],
    episodes: npt.NDArray[np.int64],
    all_rewards: Iterable[npt.NDArray[np.float64]],
    all_steps: Iterable[npt.NDArray[np.float64]],
    savefig_folder: Path | None = None,
    show: bool = True,
):
    """Plot the steps and rewards from dataframes."""
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))

    for ms, rewards, steps in zip(map_sizes, all_rewards, all_steps):
        s_mean, s_lb, s_ub = calc_std_bounds(steps, clip=(0, np.inf))
        axes[0].plot(episodes, s_mean, label=f"map size: {ms}x{ms}")
        axes[0].fill_between(episodes, s_lb, s_ub, alpha=0.6)
        axes[0].set(ylabel="Averaged steps number per Run")
        axes[0].legend()

        r_mean, r_lb, r_ub = calc_std_bounds(rewards.cumsum(axis=1), clip=(0, np.inf))
        axes[1].plot(episodes, r_mean, label=f"map size: {ms}x{ms}")
        axes[1].fill_between(episodes, r_lb, r_ub, alpha=0.6)
        axes[1].set(ylabel="Cumulated rewards per Run", xlabel="Episodes")
        axes[1].legend()

    fig.tight_layout()
    if savefig_folder:
        img_title = "frozenlake_steps_and_rewards_different_map_sizes.png"
        fig.savefig(savefig_folder / img_title, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig
