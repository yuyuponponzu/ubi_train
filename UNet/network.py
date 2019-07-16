#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from chainer import Chain, serializers, optimizers, cuda, config
import chainer.links as L
import chainer.functions as F
import numpy as np
import const
from librosa.core import load, resample, stft
from librosa.util import find_files
from librosa.effects import pitch_shift, time_stretch
import os


cp = cuda.cupy


class UNet(Chain):
    def __init__(self):
        super(UNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 16, 4, 2, 1)
            self.norm1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(16, 32, 4, 2, 1)
            self.norm2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(32, 64, 4, 2, 1)
            self.norm3 = L.BatchNormalization(64)
            self.conv4 = L.Convolution2D(64, 128, 4, 2, 1)
            self.norm4 = L.BatchNormalization(128)
            self.conv5 = L.Convolution2D(128, 256, 4, 2, 1)
            self.norm5 = L.BatchNormalization(256)
            self.conv6 = L.Convolution2D(256, 512, 4, 2, 1)
            self.norm6 = L.BatchNormalization(512)
            self.deconv1 = L.Deconvolution2D(512, 256, 4, 2, 1)
            self.denorm1 = L.BatchNormalization(256)
            self.deconv2 = L.Deconvolution2D(512, 128, 4, 2, 1)
            self.denorm2 = L.BatchNormalization(128)
            self.deconv3 = L.Deconvolution2D(256, 64, 4, 2, 1)
            self.denorm3 = L.BatchNormalization(64)
            self.deconv4 = L.Deconvolution2D(128, 32, 4, 2, 1)
            self.denorm4 = L.BatchNormalization(32)
            self.deconv5 = L.Deconvolution2D(64, 16, 4, 2, 1)
            self.denorm5 = L.BatchNormalization(16)
            self.deconv6 = L.Deconvolution2D(32, 1, 4, 2, 1)

    def __call__(self, X):

        h1 = F.leaky_relu(self.norm1(self.conv1(X)))
        h2 = F.leaky_relu(self.norm2(self.conv2(h1)))
        h3 = F.leaky_relu(self.norm3(self.conv3(h2)))
        h4 = F.leaky_relu(self.norm4(self.conv4(h3)))
        h5 = F.leaky_relu(self.norm5(self.conv5(h4)))
        h6 = F.leaky_relu(self.norm6(self.conv6(h5)))
        dh = F.relu(F.dropout(self.denorm1(self.deconv1(h6))))
        dh = F.relu(F.dropout(self.denorm2(self.deconv2(F.concat((dh, h5))))))
        dh = F.relu(F.dropout(self.denorm3(self.deconv3(F.concat((dh, h4))))))
        dh = F.relu(self.denorm4(self.deconv4(F.concat((dh, h3)))))
        dh = F.relu(self.denorm5(self.deconv5(F.concat((dh, h2)))))
        dh = F.sigmoid(self.deconv6(F.concat((dh, h1))))
        return dh

    def load(self, fname="unet.model"):
        serializers.load_npz(fname, self)

    def save(self, fname="unet.model"):
        serializers.save_npz(fname, self)


class UNetTrainmodel(Chain):
    def __init__(self, unet):
        super(UNetTrainmodel, self).__init__()
        with self.init_scope():
            self.unet = unet

    def __call__(self, X, Y):
        O = self.unet(X)
        self.loss = F.mean_absolute_error(X*O, Y)
        return self.loss

def Savespec(y_inv, y_noi, X, Y, idx_item, i, cnt, ratio = [1, 0.5]):
    S_inv = np.abs(
        stft(np.array(y_inv, dtype = 'float32'), n_fft=const.FFT_SIZE, hop_length=const.H)).astype(np.float32)
    S_noi = np.abs(
        stft(np.array(y_noi, dtype = 'float32'), n_fft=const.FFT_SIZE, hop_length=const.H)).astype(np.float32)
    norm = S_inv.max()
    S_inv /= norm
    S_noi /= norm

    for j in ratio:
        S_mix = np.maximum(0, S_inv + S_noi * j)
        S_mix /= norm
        randidx = np.random.randint(len(S_mix[1])-const.PATCH_LENGTH-1)
        X[i + cnt, 0, :, :] = \
                S_mix[1:, randidx:randidx+const.PATCH_LENGTH]
        Y[i + cnt, 0, :, :] = \
                S_inv[1:, randidx:randidx+const.PATCH_LENGTH]
        cnt += 1
    return X, Y, cnt

def Augmentation(y_inv, y_noi, X, Y, idx_item, i):
    cnt = 0
    X, Y, cnt = Savespec(y_inv, y_noi, X , Y, idx_item, i, cnt, ratio = [0.5, 0.75, 1.0, 1.25]) #ratio=[1+k/10 for k in range(-2, 2)])
    #一対のデータに対して40(shiftのみ)+80(stretch含め)=120個のデータを作成
    st_ = [1.5, 0.75]
    for sh in range(1,5):
        for inv_sh in range(1,2):#1,3
            y_inv_shift = pitch_shift(y_inv, const.SR, inv_sh)
            y_noi_shift = pitch_shift(y_noi, const.SR, sh)
            X, Y, cnt = Savespec(y_inv_shift, y_noi_shift, X , Y, idx_item, i, cnt)
            # 反転したものも
            X, Y, cnt = Savespec(np.flip(y_inv_shift), \
                np.flip(y_noi_shift), X , Y, idx_item, i, cnt, ratio = [1])
            X, Y, cnt = Savespec(y_inv_shift, np.flip(y_noi_shift),\
                X , Y, idx_item, i, cnt, ratio = [1])
            X, Y, cnt = Savespec(np.flip(y_inv_shift), \
                y_noi_shift, X , Y, idx_item, i, cnt, ratio = [1])

            for st in st_:
                y_inv_stretch = time_stretch(y_inv_shift, st)
                y_noi_stretch = time_stretch(y_noi_shift, st)
                X, Y, cnt = Savespec(y_inv_stretch, y_noi_stretch, X , Y, idx_item, i, cnt)

            y_inv_shift = pitch_shift(y_inv, const.SR, -inv_sh)
            y_noi_shift = pitch_shift(y_noi, const.SR, -sh)
            X, Y, cnt = Savespec(y_inv_shift, y_noi_shift, X , Y, idx_item, i, cnt)

            for st in st_:
                y_inv_stretch = time_stretch(y_inv_shift, st)
                y_noi_stretch = time_stretch(y_noi_shift, st)
                X, Y, cnt = Savespec(y_inv_stretch, y_noi_stretch, X , Y, idx_item, i, cnt)
    return X, Y

def Augmentation_main(inverter_, fl_noise, X, Y, idx_item, i):
    inverter_ = inverter_[np.nonzero(inverter_)[0][0]:]
    idx_item_noise = np.random.randint(0, len(fl_noise), 1)
    y_noise, sr = load(fl_noise[idx_item_noise[0]], sr=None)
    y_noise = y_noise[np.nonzero(y_noise)[0][0]:]
    minlength = 441000
    if minlength > len(y_noise):
        y_noise = np.pad(y_noise,[(0,441000-len(y_noise))],"constant")
    elif minlength > len(inverter_):
        inverter_ = np.pad(inverter_,[(0,441000-len(inverter_))],"constant")
    y_inverter = resample(inverter_[:minlength], 44100, const.SR)
    y_noise = resample(y_noise[:minlength], 44100, const.SR)
    print(len(y_noise))
    print(len(y_inverter))
    return Augmentation(y_inverter, y_noise, X, Y, idx_item, i)

def TrainUNet(PATH_inverter = "Noise", PATH_noise = "Inverter", epoch=500, savefile="unet.model"):
    unet = UNet()
    model = UNetTrainmodel(unet)
    model.to_gpu(0)
    opt = optimizers.Adam()
    opt.setup(model)
    config.train = True
    config.enable_backprop = True
    fl_noise = find_files(PATH_noise, ext="wav")
    fl_inverter = find_files(PATH_inverter, ext="wav")
    itemcnt = len(fl_inverter)# * len(fl_noise) * const.AUG_SIZE
    itemlength = 10 * itemcnt * len(fl_noise) * const.AUG_SIZE
    #ミニバッチによる1epochあたり必要な学習数に加えて
    #時間領域でのランダムサンプリングを行うことによる必要な最低回数も考慮
    subepoch = \
         itemlength // const.PATCH_LENGTH // (const.BATCH_SIZE // const.AUG_SIZE) * 4
    for ep in range(epoch):
        sum_loss = 0.0
        for subep in range(subepoch):
            X = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                         dtype="float32")
            Y = np.zeros((const.BATCH_SIZE, 1, 512, const.PATCH_LENGTH),
                         dtype="float32")
            idx_item = np.random.randint(0, itemcnt, const.BATCH_SIZE // const.AUG_SIZE)
            for i in range(const.BATCH_SIZE // const.AUG_SIZE):
                inverter_, _ = load(fl_inverter[idx_item[i]], sr=None)
                X, Y = Augmentation_main(inverter_, fl_noise, X, Y, idx_item, i*const.AUG_SIZE)
            opt.update(model, cp.asarray(X), cp.asarray(Y))
            sum_loss += model.loss.data * const.BATCH_SIZE

        print("epoch: %d/%d  loss=%.3f" % (ep+1, epoch, sum_loss))

    unet.save(savefile)