#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini, Lukas Burget, Mireia Diez)
# Copyright 2022 AUDIAS Universidad Autonoma de Madrid (author: Alicia Lozano-Diez)
# Licensed under the MIT license.

from itertools import permutations
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch.nn.functional import logsigmoid
from scipy.optimize import linear_sum_assignment


def pit_loss_multispk(
        logits: List[torch.Tensor], target: List[torch.Tensor],
        n_speakers: np.ndarray, detach_attractor_loss: bool):
    if detach_attractor_loss:
        # -1's for speakers that do not have valid attractor
        for i in range(target.shape[0]):
            target[i, :, n_speakers[i]:] = -1 * torch.ones(
                          target.shape[1], target.shape[2]-n_speakers[i])

    logits_t = logits.detach().transpose(1, 2)
    cost_mxs = -logsigmoid(logits_t).bmm(target) - logsigmoid(-logits_t).bmm(1-target)

    max_n_speakers = max(n_speakers)

    for i, cost_mx in enumerate(cost_mxs.cpu().numpy()):
        if max_n_speakers > n_speakers[i]:
            max_value = np.absolute(cost_mx).sum()
            cost_mx[-(max_n_speakers-n_speakers[i]):] = max_value
            cost_mx[:, -(max_n_speakers-n_speakers[i]):] = max_value
        pred_alig, ref_alig = linear_sum_assignment(cost_mx)
        assert (np.all(pred_alig == np.arange(logits.shape[-1])))
        target[i, :] = target[i, :, ref_alig]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
             logits, target, reduction='none')

    loss[torch.where(target == -1)] = 0
    # normalize by sequence length
    loss = torch.sum(loss, axis=1) / (target != -1).sum(axis=1)
    for i in range(target.shape[0]):
        loss[i, n_speakers[i]:] = torch.zeros(loss.shape[1]-n_speakers[i])

    # normalize in batch for all speakers
    loss = torch.mean(loss)
    return loss


def vad_loss(ys: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    # Take from reference ts only the speakers that do not correspond to -1
    # (-1 are padded frames), if the sum of their values is >0 there is speech
    vad_ts = (torch.sum((ts != -1)*ts, 2, keepdim=True) > 0).float()
    # We work on the probability space, not logits. We use silence probabilities
    ys_silence_probs = 1-torch.sigmoid(ys)
    # The probability of silence in the frame is the product of the
    # probability that each speaker is silent
    silence_prob = torch.prod(ys_silence_probs, 2, keepdim=True)
    # Estimate the loss. size=[batch_size, num_frames, 1]
    loss = F.binary_cross_entropy(silence_prob, 1-vad_ts, reduction='none')
    # "torch.max(ts, 2, keepdim=True)[0]" keeps the maximum along speaker dim
    # Invalid frames in the sequence (padding) will be -1, replace those
    # invalid positions by 0 so that those losses do not count
    loss[torch.where(torch.max(ts, 2, keepdim=True)[0] < 0)] = 0
    # normalize by sequence length
    # "torch.sum(loss, axis=1)" gives a value per batch
    # if torch.mean(ts,axis=2)==-1 then all speakers were invalid in the frame,
    # therefore we should not account for it
    # ts is size [batch_size, num_frames, num_spks]
    loss = torch.sum(loss, axis=1) / (torch.mean(ts, axis=2) != -1).sum(axis=1, keepdims=True)
    # normalize in batch for all speakers
    loss = torch.mean(loss)
    return loss


def standard_loss(ys: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(ys, ts, reduction='none')
    # zero parts of sequences that correspond to padding
    loss[torch.where(ts == -1)] = 0
    # normalize by sequence length
    loss = torch.sum(loss, axis=1) / (ts != -1).sum(axis=1)
    # normalize in batch for all speakers
    loss = torch.mean(loss)
    return loss
