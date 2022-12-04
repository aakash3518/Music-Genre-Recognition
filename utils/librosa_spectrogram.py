#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 20:16:52 2022

@author: anand
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def display_librosa(filename):
    y, sr = librosa.load(filename)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure()
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    librosa.display.cmap(Xdb)
    plt.colorbar()
    plt.title(filename)
    plt.show()
    mel_spectrogram = librosa.feature.melspectrogram\
        (y=y, sr=sr, n_fft=4096, hop_length=512, n_mels=10, fmax=10000)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    plt.figure()
    librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=sr)
    librosa.display.cmap(log_mel_spectrogram)
    plt.colorbar()
    plt.title(filename)
    plt.show()
    
    fig, ax = plt.subplots(nrows=2, sharex=True)
    chromagram = librosa.feature.chroma_stft(y=y, sr=22050, S=None, n_fft=2048, \
            hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', tuning=None, n_chroma=12)
    
    librosa.display.specshow(chromagram, x_axis="time", y_axis="chroma", sr=sr, ax=ax[0])
    librosa.display.cmap(chromagram)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)    
    times = librosa.times_like(onset_env, sr=sr, hop_length=512)
    ax[0].label_outer()
    ax[0].set(title=filename)
    ax[1].plot(times, librosa.util.normalize(onset_env), label='Onset strength')
    ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',linestyle='--', label='Beats')
    ax[1].legend()
    plt.show()


if __name__ == '__main__':
    path = 'data/raw/Progressive_Rock_Songs/01 - Birds of Fire.mp3'
    # Yes, Close to the Edge, keyboard solo
    filename = "01 - Birds of Fire.mp3"
    display_librosa(path)
