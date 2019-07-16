#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import util

"""
学習
"""
import network

#Xlist,Ylist = util.LoadDataset(target="vocal")
#print("Dataset loaded.")
network.TrainUNet()

"""
音源分離
"""

"""
fname = "original_mix.wav"
mag, phase = util.LoadAudio(fname)
start = 2048
end = 2048+1024

mask = util.ComputeMask(mag[:, start:end], unet_model="unet.model", hard=False)

util.SaveAudio(
    "inverter-%s" % fname, mag[:, start:end]*mask, phase[:, start:end])
util.SaveAudio(
    "noise-%s" % fname, mag[:, start:end]*(1-mask), phase[:, start:end])
util.SaveAudio(
    "orig-%s" % fname, mag[:, start:end], phase[:, start:end])
"""