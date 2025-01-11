from data.base import TurnType
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import os

import logging


def get_logger(namespace=__name__, level=logging.INFO):
    logger = logging.getLogger(namespace)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_abs_path(filepath):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)


def get_new_filename(save_dir):
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d:%H-%M-%S")
    return os.path.join(save_dir, current_time_str)


def get_latest_model(path):
    list_dir = os.listdir(path)
    latest_model = None
    latest_index = -1
    for item in list_dir:
        if item[:5] == "model":
            index = int("".join(x for x in item if x.isdigit()))
            if latest_model is None or index > latest_index:
                latest_index = index
                latest_model = item

    if latest_model is None:
        raise RuntimeError("model file not found")
    return os.path.join(path, latest_model)


def get_turn_after_bc(
    tokensA, tokensB, eot_items, typesA=None, typesB=None, eint=-1, ebc=-1
):
    bc_maskA = torch.zeros_like(tokensA)
    bc_maskB = torch.zeros_like(tokensB)

    eot_overlapsA = []
    eot_overlapsB = []

    eotA = torch.isin(tokensA, eot_items)
    eotB = torch.isin(tokensB, eot_items)

    overlapA = torch.logical_or(torch.eq(tokensA, eint), torch.eq(tokensA, ebc))
    overlapB = torch.logical_or(torch.eq(tokensB, eint), torch.eq(tokensB, ebc))
    if typesA is not None and typesB is not None:
        overlapA = torch.logical_or(
            tokensA == TurnType.INTERRUPT, tokensA == TurnType.BACKCHANNEL
        )
        overlapB = torch.logical_or(
            tokensB == TurnType.INTERRUPT, tokensB == TurnType.BACKCHANNEL
        )
        shifted_overlapA = torch.roll(overlapA, shifts=-1, dims=-1)
        shifted_overlapB = torch.roll(overlapB, shifts=-1, dims=-1)

        # Boundary on edge of turn (last token in BC should be 1 and inside turn should be 0)
        # Token prior to turn is -1
        boundaryA = shifted_overlapA - overlapA
        boundaryB = shifted_overlapB - overlapB

        overlapA = (boundaryA == 1).long()
        overlapB = (boundaryB == 1).long()

    cumsum_overlapA = torch.cumsum(overlapA, dim=-1)
    cumsum_overlapB = torch.cumsum(overlapB, dim=-1)

    new_overlapA = cumsum_overlapA
    new_overlapB = cumsum_overlapB

    # Each turn end has an ID corresponding to a BC
    first_eotA = new_overlapB * eotA
    first_eotB = new_overlapA * eotB

    # Extract active turn regions
    for eot in first_eotA.nonzero():
        if len(overlapB) == 0:
            continue

        overlap = overlapB.nonzero(as_tuple=True)
        batch_overlap = overlap[0]
        seq_overlap = overlap[1]
        if len(eot_overlapsA) == 0 or eot[0] != eot_overlapsA[-1][0]:
            overlap = overlap[1][(seq_overlap < eot[1]) & (batch_overlap == eot[0])]
            if len(overlap) == 0:
                continue
            overlap = overlap[-1]

            eot_overlapsA.append((eot[0], overlap, eot[1]))

            mask = (eot[0], torch.arange(overlap, eot[1] + 1))
            bc_maskA[mask] = torch.tensor(1, device="cuda")

            continue
        elif overlapB[eot[0], eot_overlapsA[-1][-1] : eot[1]].sum() > 0:
            overlap = overlap[1][
                (batch_overlap == eot[0])
                & (seq_overlap > eot_overlapsA[-1][2])
                & (seq_overlap < eot[1])
            ]
            if len(overlap) == 0:
                continue
            overlap = overlap[-1]

            eot_overlapsA.append((eot[0], overlap, eot[1]))

            mask = (eot[0], torch.arange(overlap, eot[1] + 1))
            bc_maskA[mask] = torch.tensor(1, device="cuda")

    for eot in first_eotB.nonzero():
        if len(overlapA) == 0:
            continue

        overlap = overlapA.nonzero(as_tuple=True)
        batch_overlap = overlap[0]
        seq_overlap = overlap[1]
        if len(eot_overlapsB) == 0 or eot[0] != eot_overlapsB[-1][0]:
            overlap = overlap[1][(seq_overlap < eot[1]) & (batch_overlap == eot[0])]
            if len(overlap) == 0:
                continue
            overlap = overlap[-1]

            eot_overlapsB.append((eot[0] + 1, overlap, eot[1]))

            mask = (eot[0], torch.arange(overlap, eot[1] + 1))
            bc_maskB[mask] = torch.tensor(1, device="cuda")
            continue
        elif overlapA[eot[0], eot_overlapsB[-1][-1] : eot[1]].sum() > 0:
            overlap = overlap[1][
                (batch_overlap == eot[0])
                & (seq_overlap > eot_overlapsB[-1][2])
                & (seq_overlap < eot[1])
            ]
            if len(overlap) == 0:
                continue

            overlap = overlap[-1]
            eot_overlapsB.append((eot[0], overlap, eot[1]))

            mask = (eot[0], torch.arange(overlap, eot[1] + 1))
            bc_maskB[mask] = torch.tensor(1, device="cuda")

    return bc_maskA, bc_maskB


def plot_trp(
    trp,
    text=None,
    eos_token="[SEP]",
    special_tokens=[],
    figsize=(18, 6),
    plot=False,
    fig=None,
    ax=None,
    max_idx_plot=-1,
    show=False,
    title="NONE",
):
    ax2 = None
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        ax = ax1

    special_tokens += [eos_token]
    ax.set_title(title)

    colors = ["r", "g", "b", "c", "m", "y", "k"]
    for idx, prob in enumerate(trp):
        x = torch.arange(len(trp[0]))
        offset = (idx - len(trp) // 2) * 0.3
        ax.bar(
            x + offset, prob, width=0.3, color=colors[idx], label=special_tokens[idx]
        )
        ax.set_xticks(x)

    ax.set_xticklabels(text, rotation=60)
    for idx, text in enumerate(text):
        if text == eos_token:
            ax.vlines(idx, ymin=0, ymax=1, linestyle="dashed", color="r")
        if idx == max_idx_plot:
            ax.vlines(idx, ymin=0, ymax=1, linestyle="solid", color="g")

    ax.set_ylim([0, 1])
    ax.legend()
    fig.tight_layout()

    if plot:
        plt.pause(0.01)

    if show:
        plt.show()

    return fig, (ax, ax2)


def plot_dual_trp(
    trpA,
    trpB,
    text=None,
    eos_token="[SEP]",
    special_tokens=[],
    figsize=(18, 6),
    plot=False,
    fig=None,
    ax=None,
    max_idx_plot=-1,
    show=False,
    labels={"Serialised", "Fully Aligned"},
):
    ax2 = None
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        ax = ax1

    colors = ["r", "g", "b", "c", "m", "y", "k"]
    for idx, prob in enumerate(trpA):
        x = torch.arange(len(trpA[0]))
        offset = (idx - len(trpA) // 2) * 0.3
        ax.bar(
            x + offset,
            prob,
            width=0.3,
            color=colors[idx],
            label=f"{labels[0]}:{special_tokens[idx]}",
        )
        ax.set_xticks(x)
    for idx, prob in enumerate(trpA):
        x = torch.arange(len(trpA[0]))
        offset = (idx - len(trpA) // 2) * 0.3
        ax.bar(
            x - offset,
            prob,
            width=0.3,
            color=colors[idx],
            label=f"{labels[1]}:{special_tokens[idx]}",
        )
        ax.set_xticks(x)

    ax.set_xticklabels(text, rotation=60)
    for idx, text in enumerate(text):
        if text == eos_token:
            ax.vlines(idx, ymin=0, ymax=1, linestyle="dashed", color="r")
        if idx == max_idx_plot:
            ax.vlines(idx, ymin=0, ymax=1, linestyle="solid", color="g")

    ax.set_ylim([0, 1])
    ax.legend()
    fig.tight_layout()

    if plot:
        plt.pause(0.01)

    if show:
        plt.show()

    return fig, (ax, ax2)
