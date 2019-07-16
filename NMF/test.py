from nmf import SSNMF
import numpy as np

# 動作確認。Y と FG + HU がある程度近似的に見えれば OK
# このパラメータでは近似度がまだ低く、コスト関数の値をみてもまだまだ収束していない感がある
# SSNMF のパラメータ引数 n_iter を大きくすると計算量が増えるが、近似度が高くなる。

np.random.seed(1)
comps = np.array(((1,0), (0,1), (1,1)))
activs = np.array(((0,0,1,0,1,5,0,7,9,6,5,0), (2,1,0,1,1,2,1,0,0,0,6,0)))
Y = np.dot(comps, activs)

print('original data\n---------------')
print('components:\n', comps)
print('activations:\n', activs)
print('Y:\n', Y)

computed = SSNMF(Y, F=np.array(((1,0,1),)).T, R=2, beta=0, p=True)

print('\ndecomposed\n---------------')
print('F:\n', computed[0])
print('G:\n', computed[1])
print('H:\n', computed[2])
print('U:\n', computed[3])
print('FG + HU:\n', np.dot(computed[0], computed[1]) + np.dot(computed[2], computed[3]))
print('cost:\n', computed[4])
