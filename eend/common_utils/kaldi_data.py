#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

# This library provides utilities for kaldi-style data directory.


import io
import numpy as np
import os
import soundfile as sf
import subprocess
import sys
from functools import lru_cache
from typing import Any, Dict, Tuple


def load_segments_hash(segments_file):
    ret = {}
    if not os.path.exists(segments_file):
        return None
    for line in open(segments_file):
        utt, rec, st, et = line.strip().split()
        ret[utt] = (rec, float(st), float(et))
    return ret


def load_segments_rechash(segments_file: str) -> Dict[str, Dict[str, Any]]:
    ret = {}
    if not os.path.exists(segments_file):
        return None
    for line in open(segments_file):
        utt, rec, st, et = line.strip().split()
        if rec not in ret:
            ret[rec] = []
        ret[rec].append({'utt': utt, 'st': float(st), 'et': float(et)})
    return ret


def load_wav_scp(wav_scp_file: str) -> Dict[str, str]:
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
    return {x[0]: x[1] for x in lines}


@lru_cache(maxsize=1)
def load_wav(
    wav_rxfilename: str,
    start: int,
    end: int
) -> Tuple[np.ndarray, int]:
    """ This function reads audio file and return data in numpy.float32 array.
        "lru_cache" holds recently loaded audio so that can be called
        many times on the same audio file.
        OPTIMIZE: controls lru_cache size for random access,
        considering memory size
    """
    if wav_rxfilename.endswith('|'):
        # input piped command
        p = subprocess.Popen(wav_rxfilename[:-1], shell=True,
                             stdout=subprocess.PIPE)
        data, samplerate = sf.read(io.BytesIO(p.stdout.read()),
                                   dtype='float32')
        # cannot seek
        data = data[start:end]
    elif wav_rxfilename == '-':
        # stdin
        data, samplerate = sf.read(sys.stdin, dtype='float32')
        # cannot seek
        data = data[start:end]
    else:
        # normal wav file
        data, samplerate = sf.read(wav_rxfilename, start=start, stop=end)
    return data, samplerate


def load_utt2spk(utt2spk_file: str) -> Dict[str, str]:
    """ returns dictionary { uttid: spkid } """
    lines = [line.strip().split(None, 1) for line in open(utt2spk_file)]
    return {x[0]: x[1] for x in lines}


def load_spk2utt(spk2utt_file: str) -> Dict[str, str]:
    """ returns dictionary { spkid: list of uttids } """
    if not os.path.exists(spk2utt_file):
        return None
    lines = [line.strip().split() for line in open(spk2utt_file)]
    return {x[0]: x[1:] for x in lines}


def load_reco2dur(reco2dur_file: str) -> Dict[str, float]:
    """ returns dictionary { recid: duration }  """
    if not os.path.exists(reco2dur_file):
        return None
    lines = [line.strip().split(None, 1) for line in open(reco2dur_file)]
    return {x[0]: float(x[1]) for x in lines}


class KaldiData:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.segments = load_segments_rechash(
                os.path.join(self.data_dir, 'segments'))
        self.utt2spk = load_utt2spk(
                os.path.join(self.data_dir, 'utt2spk'))
        self.wavs = load_wav_scp(
                os.path.join(self.data_dir, 'wav.scp'))
        self.reco2dur = load_reco2dur(
                os.path.join(self.data_dir, 'reco2dur'))
        self.spk2utt = load_spk2utt(
                os.path.join(self.data_dir, 'spk2utt'))

    def load_wav(
        self,
        recid: str,
        start: int,
        end: int
    ) -> Tuple[np.ndarray, int]:
        data, rate = load_wav(
            self.wavs[recid], start, end)
        return data, rate
