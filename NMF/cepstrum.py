import wave
import numpy as np

def FilterBank(fs, nfft, numChannels, fmax, fmin):
    df = fs / nfft 
    # 周波数インデックスの最大数
    nmax = int(fmax / df)
    nmin = int(fmin / df)
    # メル尺度における各フィルタの中心周波数を求める
    dmel = (fmax-fmin) / (numChannels+1)
    fcenters = np.arange(int(fmin+dmel), fmax, dmel) 
    # 各フィルタの中心周波数を周波数インデックスに変換
    indexcenter = np.round(fcenters / df)
    # 各フィルタの開始位置のインデックス
    indexstart = np.hstack(([nmin], indexcenter[0:numChannels - 1]))
    # 各フィルタの終了位置のインデックス
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))
    filterbank = np.zeros((numChannels, int(nfft/2)))#要素0のnumChannels行nmax列の配列作成
    for c in np.arange(0, numChannels):
        # 三角フィルタの左の直線の傾きから点を求める
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):    
            i = int(i)
            filterbank[c, i] = (i - indexstart[c]) * increment
        # 三角フィルタの右の直線の傾きから点を求める
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            i = int(i)
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)
    return filterbank, fcenters

    