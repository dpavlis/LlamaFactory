# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from typing import Any

from transformers.trainer import TRAINER_STATE_NAME

from . import logging
from .packages import is_matplotlib_available


if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker


logger = logging.get_logger(__name__)

DEFAULT_LOSS_PLOT_DPI = 300


def smooth(scalars: list[float]) -> list[float]:
    r"""EMA implementation according to TensorBoard."""
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def _get_step(log: dict[str, Any]) -> int | None:
    if "current_steps" in log:
        return log["current_steps"]

    if "step" in log:
        return log["step"]

    return None


def _collect_series(trainer_log: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    steps, values = [], []
    for log in trainer_log:
        if key in log:
            step = _get_step(log)
            if step is None:
                continue

            steps.append(step)
            values.append(log[key])

    return steps, values


def _format_metric_value(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _annotate_latest_value(
    ax: Any,
    steps: list[int],
    values: list[float],
    prefix: str,
    color: str,
    y_offset: int,
) -> None:
    if not steps or not values:
        return

    ax.annotate(
        f"{prefix}: {_format_metric_value(values[-1])}",
        xy=(steps[-1], values[-1]),
        xytext=(-8, y_offset),
        textcoords="offset points",
        color=color,
        fontsize=9,
        ha="right",
        va="bottom" if y_offset >= 0 else "top",
        annotation_clip=False,
    )


def gen_loss_plot(trainer_log: list[dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""Plot loss curves in LlamaBoard."""
    plt.close("all")
    plt.switch_backend("agg")
    # Keep the same default figure size but render more pixels for sharper popup zoom.
    fig = plt.figure(dpi=int(os.getenv("LLAMABOARD_LOSS_PLOT_DPI", str(DEFAULT_LOSS_PLOT_DPI))))
    ax = fig.add_subplot(111)

    train_steps, train_losses = _collect_series(trainer_log, "loss")
    eval_steps, eval_losses = _collect_series(trainer_log, "eval_loss")
    if not eval_losses:
        eval_steps, eval_losses = _collect_series(trainer_log, "eval/loss")

    if train_losses:
        if len(train_losses) == 1:
            ax.plot(
                train_steps,
                train_losses,
                color="#1f77b4",
                marker="o",
                markersize=7,
                linestyle="None",
                label="train (raw)",
            )
            ax.plot(
                train_steps,
                smooth(train_losses),
                color="#1f77b4",
                marker="o",
                markersize=5,
                linestyle="None",
                label="train (smoothed)",
            )
        else:
            ax.plot(train_steps, train_losses, color="#1f77b4", alpha=0.35, label="train (raw)")
            ax.plot(train_steps, smooth(train_losses), color="#1f77b4", label="train (smoothed)")

    if eval_losses:
        if len(eval_losses) == 1:
            ax.plot(
                eval_steps,
                eval_losses,
                color="#ff7f0e",
                marker="o",
                markersize=7,
                linestyle="None",
                label="eval (raw)",
            )
            ax.plot(
                eval_steps,
                smooth(eval_losses),
                color="#ff7f0e",
                marker="o",
                markersize=5,
                linestyle="None",
                label="eval (smoothed)",
            )
        else:
            ax.plot(eval_steps, eval_losses, color="#ff7f0e", alpha=0.35, label="eval (raw)")
            ax.plot(eval_steps, smooth(eval_losses), color="#ff7f0e", label="eval (smoothed)")

    _annotate_latest_value(ax, train_steps, train_losses, "train loss", "#1f77b4", 8)
    _annotate_latest_value(ax, eval_steps, eval_losses, "eval loss", "#ff7f0e", -8)

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    return fig


def plot_loss(save_dictionary: str, keys: list[str] = ["loss"]) -> None:
    r"""Plot loss curves and saves the image."""
    plt.switch_backend("agg")
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning_rank0(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        if steps and metrics:
            plt.annotate(
                f"{key.replace('_', ' ')}: {_format_metric_value(metrics[-1])}",
                xy=(steps[-1], metrics[-1]),
                xytext=(-8, 8),
                textcoords="offset points",
                color="#1f77b4",
                fontsize=9,
                ha="right",
                va="bottom",
                annotation_clip=False,
            )
        plt.title(f"training {key} of {save_dictionary}")
        plt.xlabel("step")
        plt.ylabel(key)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=300)
        print("Figure saved at:", figure_path)
