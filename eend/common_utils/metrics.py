#!/usr/bin/env python3

# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from typing import Dict
import torch


def calculate_metrics(
    target: torch.Tensor,
    decisions: torch.Tensor,
    threshold: float = 0.5,
    round_digits: int = 2,
) -> Dict[str, float]:
    epsilon = 1e-6
    res = {}
    decisions = (decisions > threshold).float()
    res["avg_ref_spk_qty"] = 0
    res["avg_pred_spk_qty"] = 0
    res["DER_FA"] = 0
    res["DER_miss"] = 0
    res["VAD_FA"] = 0
    res["VAD_miss"] = 0
    res["OSD_FA"] = 0
    res["OSD_miss"] = 0
    # Each sequence is processed separately as shorter sequences might need
    # masking. Each sequence counts for the errors independently and the mean
    # for the batch is returned.
    for seq_num in range(target.shape[0]):
        t_seq = target[seq_num, :, :]
        mask = (t_seq != -1)
        t_seq = torch.reshape(
            torch.masked_select(t_seq, mask), (-1, t_seq.shape[1]))
        d_seq = decisions[seq_num, :, :]
        d_seq = torch.reshape(
            torch.masked_select(d_seq, mask), (-1, d_seq.shape[1]))

        ref_spk_qty = t_seq.sum(axis=1)
        pred_spk_qty = d_seq.sum(axis=1)
        res["avg_ref_spk_qty"] += torch.mean(ref_spk_qty.double())
        res["avg_pred_spk_qty"] += torch.mean(pred_spk_qty.double())
        # active_frames has frames where at least one speaker is active
        active_frames = torch.where(ref_spk_qty != 0)[0]
        # speech_frames has #frames with speech (if n active speakers, n times)
        speech_frames = ref_spk_qty[active_frames].sum()
        # overlap_frames has frames where at least two speakers are active
        overlap_frames = torch.where(ref_spk_qty > 1)[0]

        diff_qty = pred_spk_qty - ref_spk_qty
        res["DER_FA"] += torch.round(
            100.0 * diff_qty[torch.where(diff_qty > 0)].sum() /
            (epsilon + speech_frames) * 10**round_digits
            ) / (10**round_digits)
        res["DER_miss"] += torch.round(
            -100.0 * diff_qty[torch.where(diff_qty < 0)].sum() /
            (epsilon + speech_frames) * 10**round_digits
            ) / (10**round_digits)
        # conf. error not calculated as computing all permutations is expensive
        # TODO use Hungarian algorithm?

        res["VAD_FA"] += round(
            100.0 *
            torch.where(
                ref_spk_qty[torch.where(pred_spk_qty > 0)[0]] < 1)[0].shape[0] /
            (epsilon + active_frames.shape[0]), 2)
        res["VAD_miss"] += round(
            100.0 *
            torch.where(
                pred_spk_qty[torch.where(ref_spk_qty > 0)[0]] < 1)[0].shape[0] /
            (epsilon + active_frames.shape[0]), 2)

        res["OSD_FA"] += round(
            100.0 *
            torch.where(
                ref_spk_qty[torch.where(pred_spk_qty > 1)[0]] < 2)[0].shape[0] /
            (epsilon + overlap_frames.shape[0]), 2)
        res["OSD_miss"] += round(
            100.0 *
            torch.where(
                pred_spk_qty[torch.where(ref_spk_qty > 1)[0]] < 2)[0].shape[0] /
            (epsilon + overlap_frames.shape[0]), 2)

    # Average across all sequences in batch
    for k, v in res.items():
        res[k] = v / target.shape[0]
    return res


def new_metrics() -> Dict[str, float]:
    metrics = {}
    for k in [
        'loss',
        'loss_standard',
        'loss_attractor',
        'avg_ref_spk_qty',
        'avg_pred_spk_qty',
        'DER_FA',
        'DER_miss',
        'VAD_FA',
        'VAD_miss',
        'OSD_FA',
        'OSD_miss'
    ]:
        metrics[k] = 0.0
    return metrics


def reset_metrics(acum_dict: Dict[str, float]) -> Dict[str, float]:
    for k in acum_dict.keys():
        acum_dict[k] = 0.0
    return acum_dict


def update_metrics(
    acum_dict: Dict[str, float],
    new_dict: Dict[str, float]
) -> Dict[str, float]:
    for k in new_dict.keys():
        assert (k in acum_dict), \
            f"The key {k} is not defined in the dictionary \
            where metrics are accumulated."
        acum_dict[k] += new_dict[k]
    return acum_dict
