#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from common_utils.kaldi_data import KaldiData
from typing import Callable, Tuple
import numpy as np
import librosa


def get_labeledSTFT(
    kaldi_obj: KaldiData,
    rec: str,
    start: int,
    end: int,
    frame_size: int,
    frame_shift: int,
    n_speakers: int = None,
    use_speaker_id: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts STFT and corresponding diarization labels for
    given recording id and start/end times
    Args:
        kaldi_obj (KaldiData)
        rec (str): recording id
        start (int): start frame index
        end (int): end frame index
        frame_size (int): number of samples in a frame
        frame_shift (int): number of shift samples
        n_speakers (int): number of speakers
            if None, the value is given from data
    Returns:
        Y: STFT
            (n_frames, n_bins)-shaped np.complex64 array,
        T: label
            (n_frmaes, n_speakers)-shaped np.int32 array.
    """
    data, rate = kaldi_obj.load_wav(
        rec, start * frame_shift, end * frame_shift)
    Y = stft(data, frame_size, frame_shift)
    filtered_segments = kaldi_obj.segments[rec]
    # filtered_segments = kaldi_obj.segments[kaldi_obj.segments['rec'] == rec]
    speakers = np.unique(
        [kaldi_obj.utt2spk[seg['utt']] for seg
         in filtered_segments]).tolist()
    if n_speakers is None:
        n_speakers = len(speakers)
    T = np.zeros((Y.shape[0], n_speakers), dtype=np.int32)

    if use_speaker_id:
        all_speakers = sorted(kaldi_obj.spk2utt.keys())
        S = np.zeros((Y.shape[0], len(all_speakers)), dtype=np.int32)

    for seg in filtered_segments:
        speaker_index = speakers.index(kaldi_obj.utt2spk[seg['utt']])
        if use_speaker_id:
            all_speaker_index = all_speakers.index(
                kaldi_obj.utt2spk[seg['utt']])
        start_frame = np.rint(
            seg['st'] * rate / frame_shift).astype(int)
        end_frame = np.rint(
            seg['et'] * rate / frame_shift).astype(int)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            T[rel_start:rel_end, speaker_index] = 1
            if use_speaker_id:
                S[rel_start:rel_end, all_speaker_index] = 1

    if use_speaker_id:
        return Y, T, S
    else:
        return Y, T


def splice(Y: np.ndarray, context_size: int = 0) -> np.ndarray:
    """ Frame splicing
    Args:
        Y: feature
            (n_frames, n_featdim)-shaped numpy array
        context_size:
            number of frames concatenated on left-side
            if context_size = 5, 11 frames are concatenated.
    Returns:
        Y_spliced: spliced feature
            (n_frames, n_featdim * (2 * context_size + 1))-shaped
    """
    Y_pad = np.pad(
        Y,
        [(context_size, context_size), (0, 0)],
        'constant')
    Y_spliced = np.lib.stride_tricks.as_strided(
        np.ascontiguousarray(Y_pad),
        (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
        (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
    return Y_spliced


def stft(
    data: np.ndarray,
    frame_size: int,
    frame_shift: int
) -> np.ndarray:
    """ Compute STFT features
    Args:
        data: audio signal
            (n_samples,)-shaped np.float32 array
        frame_size: number of samples in a frame (must be a power of two)
        frame_shift: number of samples between frames
    Returns:
        stft: STFT frames
            (n_frames, n_bins)-shaped np.complex64 array
    """
    # round up to nearest power of 2
    fft_size = 1 << (frame_size - 1).bit_length()
    # HACK: The last frame is omitted
    #       as librosa.stft produces such an excessive frame
    if len(data) % frame_shift == 0:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T[:-1]
    else:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T


def subsample(
    Y: np.ndarray,
    T: np.ndarray,
    subsampling: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """ Frame subsampling
    """
    Y_ss = Y[::subsampling]
    T_ss = T[::subsampling]
    return Y_ss, T_ss


def transform(
    Y: np.ndarray,
    sampling_rate: int,
    feature_dim: int,
    transform_type: str,
    dtype: type = np.float32,
) -> np.ndarray:
    """ Transform STFT feature
    Args:
        Y: STFT
            (n_frames, n_bins)-shaped array
        transform_type:
            None, "log"
        dtype: output data type
            np.float32 is expected
    Returns:
        Y (numpy.array): transformed feature
    """
    Y = np.abs(Y)
    if transform_type.startswith('logmel'):
        n_fft = 2 * (Y.shape[1] - 1)
        mel_basis = librosa.filters.mel(sampling_rate, n_fft, feature_dim)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        if transform_type == 'logmel_meannorm':
            mean = np.mean(Y, axis=0)
            Y = Y - mean
        elif transform_type == 'logmel_meanvarnorm':
            mean = np.mean(Y, axis=0)
            Y = Y - mean
            std = np.maximum(np.std(Y, axis=0), 1e-10)
            Y = Y / std
    else:
        raise ValueError('Unknown transform_type: %s' % transform_type)
    return Y.astype(dtype)
