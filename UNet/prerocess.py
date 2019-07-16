#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
メモリに乗る量のデータに対する前処理プログラム
Augmentationを行い，任意のフォルダにnpz形式で保存
"""
from librosa.core import load, resample, stft
import numpy as np
from librosa.util import find_files
from librosa.effects import pitch_shift, time_stretch
import const as C
import os

PATH_noise = "Noise"
PATH_inverter = "Inverter"

fl_noise = find_files(PATH_noise, ext="wav")
fl_inverter = find_files(PATH_inverter, ext="wav")

def Savespec(y_inv, y_noi, fname_inv, fname_noi):
    print(y_inv.shape)
    S_inv = np.abs(
        stft(y_inv, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)
    S_noi = np.abs(
        stft(y_noi, n_fft=C.FFT_SIZE, hop_length=C.H)).astype(np.float32)
    norm = S_inv.max()
    S_inv /= norm
    S_noi /= norm

    ratio = [1, 0.75, 0.5]
    for i in ratio:
        S_mix = np.maximum(0, S_inv + S_noi * i)
        S_mix /= norm
        np.savez(os.path.join(C.PATH_FFT, fname_inv + fname_noi + str(i) + ".npz"),
                inv=S_inv, mix=S_mix, noi=S_noi)

    # y_mix = istft(S_mix*phase, hop_length=C.H, win_length=C.FFT_SIZE)
    # write_wav(os.path.join("Audiocheck", fname+".wav"), y_mix, C.SR)


def SavespecArg(y_inv, y_noi, fname_inv, fname_noi):#, shift, stretch):
    Savespec(y_inv, y_noi, fname_inv, fname_noi)
    #一対のデータに対して40(shiftのみ)+80(stretch含め)=120個のデータを作成
    st_ = [1.5, 0.75]
    for sh in range(1,11):
        for inv_sh in range(1,3):
            y_inv_shift = pitch_shift(y_inv, C.SR, inv_sh)
            y_noi_shift = pitch_shift(y_noi, C.SR, sh)
            Savespec(y_inv_shift, y_noi_shift, fname_inv, "%s_shift%d" % (fname_noi, sh))

            for st in st_:
                y_inv_stretch = time_stretch(y_inv_shift, st)
                y_noi_stretch = time_stretch(y_noi_shift, st)
                Savespec(y_inv_stretch, y_noi_stretch, fname_inv,
                         "%s_shift%d_stretch%d" % (fname_noi, sh, int(st * 10)))

            y_inv_shift = pitch_shift(y_inv, C.SR, -inv_sh)
            y_noi_shift = pitch_shift(y_noi, C.SR, -sh)
            Savespec(y_inv_shift, y_noi_shift, fname_inv, "%s_shift-%d" % (fname_noi, -sh))

            for st in st_:
                y_inv_stretch = time_stretch(y_inv_shift, st)
                y_noi_stretch = time_stretch(y_noi_shift, st)
                Savespec(y_inv_stretch, y_noi_stretch, fname_inv, 
                         "%s_shift%d_stretch%d" % (fname_noi, -sh, int(st * 10)))

#85(ノイズデータ)*70(インバータデータ) = 6000
#6000 * 120(Augmentation) * 3(SNR可変) = 216万データを生成
#(全て10秒のデータだから6000時間分のデータ)
for i in range(0, len(fl_inverter)):
    print("Processing:" + fl_inverter[i])
    y_inverter_, _ = load(fl_inverter[i], sr=None)
    y_inverter_ = y_inverter_[np.nonzero(y_inverter_)[0][0]:]
    fname_inv = fl_inverter[i].split("/")[-1].split(".")[0]
    for j in range(0, len(fl_noise)):
        print("Processing:" + fl_noise[j])
        y_noise, sr = load(fl_noise[j], sr=None)
        y_noise = y_noise[np.nonzero(y_noise)[0][0]:]
        minlength = min([y_inverter_.size, y_noise.size])
        print(minlength)
        y_inverter = resample(y_inverter_[:minlength], 44100, C.SR)
        y_noise = resample(y_noise[:minlength], 44100, C.SR)
        fname_noi = fl_noise[j].split("/")[-1].split(".")[0]
        SavespecArg(y_inverter, y_noise, fname_inv, fname_noi)