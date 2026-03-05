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


logger = logging.get_logger(__name__)


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


def gen_loss_plot(trainer_log: list[dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""Plot loss curves in LlamaBoard."""
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    train_steps, train_losses = _collect_series(trainer_log, "loss")
    eval_steps, eval_losses = _collect_series(trainer_log, "eval_loss")
    if not eval_losses:
        eval_steps, eval_losses = _collect_series(trainer_log, "eval/loss")

    if train_losses:
        ax.plot(train_steps, train_losses, color="#1f77b4", alpha=0.35, label="train (raw)")
        ax.plot(train_steps, smooth(train_losses), color="#1f77b4", label="train (smoothed)")

    if eval_losses:
        ax.plot(eval_steps, eval_losses, color="#ff7f0e", alpha=0.35, label="eval (raw)")
        ax.plot(eval_steps, smooth(eval_losses), color="#ff7f0e", label="eval (smoothed)")

    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
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
        plt.title(f"training {key} of {save_dictionary}")
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)
