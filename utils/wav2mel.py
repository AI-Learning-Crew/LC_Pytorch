'''
Convert *.wav or *.m4a into mel spectrogram\

for unit test, run:
    python utils/wav2mel.py
'''


import random
import numpy as np
import argparse
import time
from _thread import start_new_thread
import queue
from python_speech_features import logfbank
import webrtcvad
from pydub import AudioSegment
try:
    import vad_ex
except:
    from utils import vad_ex


SUPPORTED_VAD_SAMPLE_RATES = {8000, 16000, 32000, 48000}


def _ensure_supported_sample_rate(path: str, audio_bytes: bytes, sample_rate: int) -> tuple[bytes, int]:
    if sample_rate in SUPPORTED_VAD_SAMPLE_RATES:
        return audio_bytes, sample_rate

    segment = AudioSegment.from_file(path)
    segment = segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    return segment.raw_data, segment.frame_rate


def vad_process(path):
    path = str(path)
    if path.endswith('.wav'):
        audio, sample_rate = vad_ex.read_wave(path)
    elif path.endswith('.m4a'):
        audio, sample_rate = vad_ex.read_m4a(path)
    else:
        raise TypeError('Unsupported file type: {}'.format(path.split('.')[-1]))

    audio, sample_rate = _ensure_supported_sample_rate(path, audio, sample_rate)

    vad = webrtcvad.Vad(1)
    frames = vad_ex.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
    total_wav = b""
    for i, segment in enumerate(segments):
        total_wav += segment
    # Without writing, unpack total_wav into numpy [N,1] array
    # 16bit PCM format dtype=np.int16
    wav_arr = np.frombuffer(total_wav, dtype=np.int16)
    #print("read audio data from byte string. np array of shape:" + \
    #    str(wav_arr.shape))
    return wav_arr, sample_rate


def wav_to_mel(path, nfilt=40):
    path = str(path)
    '''
    Output shape: (nfilt, length)
    '''
    wav_arr, sample_rate = vad_process(path)
    #print("sample_rate:", sample_rate)
    logmel_feats = logfbank(
        wav_arr,
        samplerate=sample_rate,
        nfilt=nfilt)
    #print("created logmel feats from audio data. np array of shape:" \
    #    + str(logmel_feats.shape))
    return np.transpose(logmel_feats)
