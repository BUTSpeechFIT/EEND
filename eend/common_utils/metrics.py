#!/usr/bin/env python3

# Copyright 2022 Brno University of Technology (author: Federico Landini, Mireia Diez)
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
    # Each error is accumulated per sequence as they might need
    # different masking. Each sequence counts for the errors independently
    # and the total speech/overlap counts are acumulated.
    # Final values are estimated for the batch and returned.
    active_frames_tot = 0
    speech_frames_tot = 0
    overlap_frames_tot = 0
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
        active_frames_tot += active_frames.shape[0]
        # speech_frames has #frames with speech (if n active speakers, n times)
        speech_frames = ref_spk_qty[active_frames].sum()
        speech_frames_tot += speech_frames
        # overlap_frames has frames where at least two speakers are active
        overlap_frames = torch.where(ref_spk_qty > 1)[0]
        overlap_frames_tot += overlap_frames.shape[0]

        diff_qty = pred_spk_qty - ref_spk_qty
        res["DER_FA"] += diff_qty[torch.where(diff_qty > 0)].sum()
        res["DER_miss"] += -diff_qty[torch.where(diff_qty < 0)].sum()
        # conf. error not calculated as computing all permutations is expensive
        # TODO use Hungarian algorithm?

        res["VAD_FA"] += torch.where(ref_spk_qty[torch.where(pred_spk_qty > 0)[0]] < 1)[0].shape[0]
        res["VAD_miss"] += torch.where(pred_spk_qty[torch.where(ref_spk_qty > 0)[0]] < 1)[0].shape[0]

        res["OSD_FA"] += torch.where(ref_spk_qty[torch.where(pred_spk_qty > 1)[0]] < 2)[0].shape[0]
        res["OSD_miss"] += torch.where(pred_spk_qty[torch.where(ref_spk_qty > 1)[0]] < 2)[0].shape[0]

    # divide by the numerators estimated in the whole batch
    res["DER_FA"] = torch.round(100 * res["DER_FA"] / (epsilon + speech_frames_tot) * 10**round_digits / (10**round_digits))
    res["DER_miss"] = torch.round(100 * res["DER_miss"] / (epsilon + speech_frames_tot) * 10**round_digits / (10**round_digits))
    res["VAD_FA"] = round(100 * res["VAD_FA"] / (epsilon + active_frames_tot), 2)
    res["VAD_miss"] = round(100 * res["VAD_miss"] / (epsilon + active_frames_tot), 2)
    res["OSD_FA"] = round(100 * res["OSD_FA"] / (epsilon + overlap_frames_tot), 2)
    res["OSD_miss"] = round(100 * res["OSD_miss"] / (epsilon + overlap_frames_tot), 2)
    res["avg_ref_spk_qty"] = res["avg_ref_spk_qty"] / target.shape[0]
    res["avg_pred_spk_qty"] = res["avg_pred_spk_qty"] / target.shape[0]

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
