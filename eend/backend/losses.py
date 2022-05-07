#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from itertools import permutations
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple


def standard_loss(ys: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(ys, ts, reduction='none')
    # zero parts of sequences that correspond to padding
    loss[torch.where(ts == -1)] = 0
    # normalize by sequence length
    loss = torch.sum(loss, axis=1) / (ts != -1).sum(axis=1)
    # normalize in batch for all speakers
    loss = torch.mean(loss)
    return loss


def batch_pit_n_speaker_loss(
    device: torch.device,
    ys: torch.Tensor,
    ts: torch.Tensor,
    n_speakers_list: List[int]
) -> Tuple[float, torch.Tensor]:
    """
    PIT loss over mini-batch.
    Args:
      ys: B-length list of predictions (pre-activations)
      ts: B-length list of labels
      n_speakers_list: list of n_speakers in batch
    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    max_n_speakers = max(n_speakers_list)

    losses = []
    for shift in range(max_n_speakers):
        # rolled along with speaker-axis
        ts_roll = torch.stack([torch.roll(t, -shift, dims=1) for t in ts])
        # loss: (B, T, C)
        loss = F.binary_cross_entropy_with_logits(
            ys,
            ts_roll.float(),
            reduction='none').detach()
        # zero parts of sequences that correspond to padding
        loss[torch.where(ts_roll == -1)] = 0
        # sum over time: (B, C)
        loss = torch.sum(loss, axis=1)
        # normalize by sequence length
        loss = loss / (ts_roll != -1).sum(axis=1)
        losses.append(loss)
    # losses: (B, C, C)
    losses = torch.stack(losses, axis=2)
    # losses[b, i, j] is a loss between
    # `i`-th speaker in y and `(i+j)%C`-th speaker in t

    perms = np.asarray(list(permutations(range(max_n_speakers))), dtype="i")
    # y_ind: [0,1,2,3]
    y_ind = np.arange(max_n_speakers, dtype='i')
    #  perms  -> relation to t_inds      -> t_inds
    # 0,1,2,3 -> 0+j=0,1+j=1,2+j=2,3+j=3 -> 0,0,0,0
    # 0,1,3,2 -> 0+j=0,1+j=1,2+j=3,3+j=2 -> 0,0,1,3
    t_inds = np.mod(perms - y_ind, max_n_speakers)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(torch.mean(losses[:, y_ind, t_ind], axis=1))
    # losses_perm: (B, Perm)
    losses_perm = torch.stack(losses_perm, axis=1)

    # masks: (B, Perms)
    def select_perm_indices(num: int, max_num: int) -> List[int]:
        perms = list(permutations(range(max_num)))
        sub_perms = list(permutations(range(num)))
        return [[x[:num] for x in perms].index(perm) for perm in sub_perms]

    masks = torch.full_like(losses_perm, float("Inf"))
    for i, _ in enumerate(ts):
        n_speakers = n_speakers_list[i]
        indices = select_perm_indices(n_speakers, max_n_speakers)
        masks[i, indices] = 0
    losses_perm += masks

    # normalize across batch
    min_loss = torch.mean(torch.min(losses_perm, dim=1)[0])

    min_indices = torch.argmin(losses_perm.detach(), axis=1)
    labels_perm = [t[:, perms[idx]] for t, idx in zip(ts, min_indices)]
    labels_perm = [t[:, :n_speakers] for t, n_speakers in zip(
        labels_perm, n_speakers_list)]

    return min_loss, torch.stack(labels_perm)
