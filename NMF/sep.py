import matplotlib
#matplotlib.use('Agg')   # 画像保存用
#matplotlib.use('tkAgg')   # 画像plot用
import matplotlib.pyplot as plt
import argparse
import librosa
import librosa.display
import scipy.io.wavfile as wav
import numpy as np
import math
import sys
import tsne
import matplotlib.cm as cm
from nmf import NMF,SSNMF
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
#from sklearn.manifold import TSNE #t-SNEのsklearn実装，あまり良くないらしい


class Functions():
    def __init__(self, basis, iter, ss_basis, ss_iter):
        self.basis = basis  # NMF で分解する基底数
        self.iter = iter     # インバータ音学習で行う反復計算回数
        self.ss_basis = ss_basis  # Semi-Supervised NMF で分解する基底数
        self.ss_iter = ss_iter   # Semi-Supervised NMF で行う反復計算回数
        self.sr = 44100 # sampling rate

     # スペクトログラム を返す関数
    def stft(self, filename):
        s, sr = librosa.load(filename + ".wav")
        self.sr = sr
        stft_wav = librosa.stft(s)
        stft = np.abs(stft_wav)
        #stft_wav[0,:] = sys.float_info.epsilon
        #print(stft_wav[0,1],stft_wav[1,1])
        return stft

     # nmf = [教師の左基底行列, 教師の右アクティベーション行列, 各iterでのコスト値]
    def train_vvvf(self, stft, R=30, init_F=None, init_G=None, init='random', beta=2):
        #return NMF(stft, R=self.basis, n_iter=self.iter)
        return NMF(stft, R=R, n_iter=50, init_F=init_F, init_G=init_G, init=init, beta=beta, verbose=False)

     # ssnmf = [目的の左基底行列, 目的の右アクティベーション行列, 雑音の左基底行列, 雑音のアクティベーション行列, 各iterでのコスト値]
    def extract_vvvf(self, stft, F, beta=2, p=None):
        ssnmf = SSNMF(stft, F=F, R=self.ss_basis, n_iter=self.ss_iter, beta=beta, p=None)
        return ssnmf

     # NMFで抽出した目的音の右アクティベート行列を全て足し合わせて時間ごとの音量を求める
    def calculate_amptitudemap(self, ssnmf):
        return np.sum(ssnmf[1], axis=0)

     # ケプストラム分析により包絡を導出（対数振幅スペクトルであることに注意）
    def extract_ceps(self,stft, cepCoef = 15, nfft = 2048):
        envelope = []
        for i in range(len(stft[0])):
            one_stft = stft[:,i]
            logStft = 20 * np.log10(one_stft) #対数振幅スペクトル化
            cps = np.fft.ifft(logStft,nfft)[0:int(nfft/2)] #ケプストラムの導出
            cpsLif = [one if i <= cepCoef else 0 for i, one in enumerate(cps)] # 高周波成分を除く（左右対称なので注意）
            dftSpc = np.real(np.fft.fft(cpsLif, nfft))[0:int(nfft/2)] #ケプストラム領域をフーリエ変換してスペクトル領域に戻す
            dftSpc = dftSpc + np.ceil(-1 * np.min(dftSpc)) #正規化
            dftSpc = dftSpc / np.max(dftSpc)
            envelope.append(dftSpc) #各時間での包絡を積んでいる
        cepstrums = np.array(envelope)
        return cepstrums

    def remove_noise(self, stft, tsne_result, kmeans_tsne):
        RemNum = 2,5
        print(stft.shape)
        rem_list = np.array([])
        for i in RemNum:
            rem_list = np.concatenate([rem_list, np.where(kmeans_tsne == i)[0]], 0)
            print(np.where(kmeans_tsne == i)[0].shape)
        print(rem_list.shape)
        rem_stft = np.delete(stft, rem_list, 1)
        print(rem_stft.shape)
        return rem_stft

    def extract_ceps_plot(self, stft, cepCoef = 15, nfft = 2048):
        envelope = []
        j = 0
        plt.figure()
        for i in range(len(stft[0])):
            j += 1
            one_stft = stft[:,i]
            plt.subplot(10, 2, j)
            plt.plot(one_stft)
            #plt.title("振幅スペクトル")
            #plt.show()
            logStft = 20 * np.log10(one_stft) #対数振幅スペクトル化
            """
            plt.figure()
            plt.plot(logStft)
            plt.title("対数振幅スペクトル")
            plt.show()
            """
            cps = np.fft.ifft(logStft,nfft)[0:int(nfft/2)] #ケプストラムの導出
            """
            plt.figure()
            plt.plot(cps)
            plt.title("ケプストラム")
            plt.show()
            """
            cpsLif = [one if i <= cepCoef else 0 for i, one in enumerate(cps)] # 高周波成分を除く（左右対称なので注意）
            """
            plt.figure()
            plt.plot(cpsLif)
            plt.title("リフタ後")
            plt.show()
            """
            dftSpc = np.real(np.fft.fft(cpsLif, nfft))[0:int(nfft/2)] #ケプストラム領域をフーリエ変換してスペクトル領域に戻す
            """
            plt.figure()
            plt.plot(dftSpc)
            plt.plot(np.abs(dftSpc))
            plt.title("包絡線")
            plt.show()
            print(dftSpc[30])
            """
            dftSpc = dftSpc + np.ceil(-1 * np.min(dftSpc)) #正規化
            dftSpc = dftSpc / np.max(dftSpc)
            #plt.figure()
            j += 1
            plt.subplot(10,2,j)
            plt.plot(dftSpc)
            #plt.title("Cepstrum")
            envelope.append(dftSpc) #各時間での包絡を積んでいる
        plt.show()
        cepstrums = np.array(envelope)
        return cepstrums

     # 各時間におけるFFTの結果をplot
    def plot_fft(self, stft):
        Num = 15
        xNum = 5
        plt.figure()
        for i in range(Num):
            plt.subplot(Num, math.ceil(Num/xNum), i+1)
            cnt = int(i*110)
            one_stft = stft[:,cnt]
            plt.plot(one_stft)
            plt.title('time:'+str(cnt))
        plt.tight_layout()
        #plt.savefig('train_FFT.png')
        plt.show()

    def plot_ceps(self, cepstrums):
        Num = len(cepstrums) # cepstrums[n_samples,n_feauture]
        xNum = 5
        plt.figure()
        for i in range(Num):
            plt.subplot(xNum, math.ceil(Num/xNum), i+1)
            one_ceps = cepstrums[i,:]
            plt.plot(one_ceps)
            plt.title('No:'+str(i+1))
        #plt.savefig('train_FFT.png')
        plt.show()

     # 時間ごとの音量の変化を描画
    def plot_amptitudemap(self, ssnmf):#, filename):
        amptitudemap = self.calculate_amptitudemap(ssnmf)
        plt.figure()
        plt.plot(amptitudemap, label="default")
        plt.plot(np.convolve(amptitudemap, np.ones(100)/float(100), 'valid'), label="100")
        plt.plot(np.convolve(amptitudemap, np.ones(300)/float(300), 'valid'), label="300")
        plt.plot(np.convolve(amptitudemap, np.ones(500)/float(500), 'valid'), label="500")
        plt.plot(np.convolve(amptitudemap, np.ones(700)/float(700), 'valid'), label="700")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
        #plt.savefig(filename + '_AmplitudeMap_figure_'+str(self.basis)+'.png')
        plt.show()

     # NMFの左基底行列の画像を保存
    def plot_nmfspec(self, nmf, filename):
        left = nmf[0]
        plt.figure(figsize=(10,4))
        librosa.display.specshow(librosa.amplitude_to_db(left, ref=np.max), y_axis='log')
        plt.title('NMF Power spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        #plt.savefig(filename + '_NMF_figure_'+str(self.basis)+'.png')
        plt.show()

     # NMFの右アクティベーション行列を保存
    def plot_activation(self, nmf, filename):
        right = nmf[1]
        plt.figure()
        xs = int(math.gcd(self.basis, 10))
        for k in range(1, self.basis + 1):
            plt.subplot(xs, math.ceil(self.basis / xs), k)
            plt.plot(right[1][k-1])
            plt.title("basis:"+str(k))
        plt.tight_layout()
        #plt.savefig(filename + '_NMF_activation_'+str(self.basis)+'.png')
        plt.show()

     # trainにおける基底数ごとの教師データの復元誤差を保存
    def plot_cost(self, cnt, rem_stft):#, filename):
        left = np.arange(1, cnt+1)
        #cost = np.array([NMF(np.abs(self.stft(filename)), R=i, n_iter=self.iter) for i in range(1, cnt)])
        res = np.array([NMF(rem_stft, R=i, n_iter=self.iter) for i in range(1, cnt+1)])
        cost = [res[i][2][-1] for i in range(0, cnt)]
        plt.figure()
        plt.bar(left, cost)
        plt.title("min is {}".format(np.where(np.amin(cost) == cost)[0][0]))
        #!plt.title(filename + "min is {}".format(np.where(np.amin(cost) == cost)[0][0]))
        plt.tight_layout()
        #!plt.savefig(filename + '_NMFbasis.png')
        plt.show()
        return cost

     # NMFの左基底行列を一つずつplot
    def plot_nmf_harmonic_structure(self, nmf):#, filename):
        left = nmf
        plt.figure()
        plt.title('NMF Harmonic Structure')
        xs = int(math.gcd(self.basis, 10))
        for k in range(1, self.basis + 1):
            plt.subplot(xs, math.ceil(self.basis / xs), k)
            plt.plot(left[:,k-1])
            plt.title("basis:"+str(k))
        plt.tight_layout()
        #plt.savefig(filename + '_NMF_harmonic_structure.png')
        plt.show()

     # 計算したそれぞれのケプストラムをtsneで次元圧縮したものをplot，適切なperplexity探索用
    def plot_ceps_tsnes(self, cepstrums):
        Num = 30
        xNum = 5
        plt.figure()
        for i in range(Num):
            plt.subplot(xNum, math.ceil(Num/xNum), i+1)
            cnt = int(i*50+50)
            model = tsne.BHTSNE(dimensions=2, perplexity=cnt, theta=0.5, rand_seed=1000)
            tsne_result = model.fit_transform(cepstrums) 
            plt.plot(tsne_result[:,0], tsne_result[:,1], ".")
            plt.title("perplexity:"+str(cnt))
        #plt.savefig('train_tsnes.png')
        plt.show()

     # 探索したperplexityで計算したt-sneの結果を保存(学習に時間がかかるので)
    def save_tsne_result(self, cepstrums, fname):
        perNum = 160 # train_norm_on_cpsは100, nmf+trainは60,rem_nmf_train_cpsは160
        model = tsne.BHTSNE(dimensions=2, perplexity=perNum, theta=0.5, rand_seed=-1)
        tsne_result = model.fit_transform(cepstrums)
        np.save(fname, tsne_result)
        #np.save("./tsne_result_.npy", tsne_result)

    def calculate_clusta(self, tsne_fname, numClusta=6):
        #tsne_result = np.load("./tsne_result_nmf+train.npy") #tsneはclusta：10，pcaはclusta：
        #tsne_result = np.load("./tsne_result_.npy")
        #tsne_result = np.load("./tsne_result_nmf+train+rem.npy")
        tsne_result = np.load(tsne_fname)
        #lsa = TruncatedSVD(2)
        #tsne_result = lsa.fit_transform(cepstrums)
        #loss = round((sum(lsa.explained_variance_ratio_)),2)
        #print("説明できる割合:{}".format(loss))
        kmeans = MiniBatchKMeans(n_clusters=numClusta, max_iter=800, random_state=1000) #k-meansでクラスタリング，クラスタは見て決めた
        kmeans_tsne = kmeans.fit_predict(tsne_result) # ラベル

        return tsne_result, kmeans_tsne, kmeans

     # 探索したperplexityを使ってplotし，k-meansでクラスタリング，代表データを取得
    def plot_clusta(self, tsne_fname, numClusta=6):
        plt.figure()
        for j in range(int(numClusta)):
            clusta_leader = []
            plt.subplot(numClusta, 1, j+1)
            tsne_result, kmeans_tsne, kmeans = self.calculate_clusta(tsne_fname, j+1)
            color = cm.brg(np.linspace(0,1,np.max(kmeans_tsne) - np.min(kmeans_tsne)+1))
            for i in range(np.min(kmeans_tsne), np.max(kmeans_tsne)+1):
                plt.plot(tsne_result[kmeans_tsne == i][:,0],
                        tsne_result[kmeans_tsne == i][:,1],
                        ".",
                        color=color[i]
                        )
                ret = self._serch_neighbourhood(kmeans.cluster_centers_[i], tsne_result[kmeans_tsne == i]) #各クラスタの重心の最近傍点を探索
                plt.text(ret[0],
                        ret[1],
                        str(i+1), color="black", size=16
                        )
                clusta_leader.append(np.where(tsne_result == ret)[0][0])
                print('{}:{}'.format((i+1), clusta_leader[i])) #重心の最近傍点の添字番号をprint
        plt.show()

        return clusta_leader

        #k-meansだと繋がっているものも複数の島に分けようとするから，それが嫌ならDBSCANを使う
        """
        dbscan = DBSCAN(eps=2)
        dbscan_tsne = dbscan.fit_predict(tsne_result)
        color=cm.brg(np.linspace(0,1,np.max(dbscan_tsne) - np.min(dbscan_tsne)+1))
        for i in range(np.min(dbscan_tsne), np.max(dbscan_tsne)+1):
            plt.plot(tsne_result[dbscan_tsne == i][:,0],
                    tsne_result[dbscan_tsne == i][:,1],
                    ".",
                    color=color[i]
                    )
            plt.text(tsne_result[dbscan_tsne == i][:,0][0],
                    tsne_result[dbscan_tsne == i][:,1][0],
                    str(i), color="black", size=16
                    )
        plt.show()
        """

    def plot_clusta_leader(self, stft, cepstrums, clusta_leader):
        plt.figure()
        Num = len(clusta_leader)
        j = 0
        for i in range(Num):
            j += 1
            plt.subplot(Num,2,j)
            plt.plot(stft[:,clusta_leader[i]])
            plt.title("ClustaNumber:{},FFT".format(str(i+1)))
            j += 1
            plt.subplot(Num,2,j)
            plt.plot(cepstrums.T[:,clusta_leader[i]])
            plt.title("Cepstrum")
        plt.show()

    def calculate_coefficient(self, cepstrums, nmf_cepstrums, clusta_leader):
        cp = clusta_leader
        cp.pop(2,5) #クラスタ3と6を削除
        Num = len(cp)
        nmf_Num = len(nmf_cepstrums)
        for k in range(nmf_Num):
            j = 0
            for i in range(Num):
                j += 1
                plt.figure()
                plt.subplot(3,1,j)
                plt.plot(nmf_cepstrums.T[:,clusta_leader[i]])
                plt.title("NMF_cepstrum:basis_{}".format(k+1))
                j += 1
                plt.subplot(Num*3,3,j)
                plt.plot(cepstrums.T[:,clusta_leader[i]])
                plt.title("Number:{}, Train Cepstrum".format(str(i+1)))
                j += 1
                plt.subplot(Num*3,3,j)
                corr = np.correlate(cepstrums.T[:,clusta_leader[i]], nmf_cepstrums.T[:,clusta_leader[i]], "full") #相互相関の計算
                plt.plot(np.arange(len(corr)) - len(cepstrums.T[:,clusta_leader[i]]) + 1, corr, color="r")
                plt.title("xcorr")
            plt.show()

    def _serch_neighbourhood(self, p0, ps):
        L = np.array([])
        for i in range(ps.shape[0]):
            L = np.append(L,np.linalg.norm(ps[i]-p0))
        return ps[np.argmin(L)]


def main(args):
    func = Functions(basis=args.basis, iter=100, ss_basis=50, ss_iter=50)
    trainname = ["data/" + str(one) for one in args.train]
    targetname = ["data/" + str(one) for one in args.target]
    for train in trainname:
        for target in targetname:
            print(train,target)
            fname = "./tsne_result_nmf+train+rem.npy"
            #left_nmf = func.train_vvvf(train)[0]
            #nmf_cepstrums = func.extract_ceps_plot(left_nmf)
            #nmf_cepstrums = func.extract_ceps_plot(left_nmf)
            #STFT, ceps, tsne, remove a noise, NMF, NMF ceps, tsne, leaderの基底とceps表示
            
            """基底準備用のscript
            stft = func.stft(train)
            ceps = func.extract_ceps(stft)
            a,b,c = func.calculate_clusta("./tsne_result_.npy")
            rem_stft = func.remove_noise(stft, a,b)
            rem_ceps = func.extract_ceps(rem_stft)
            func.save_tsne_result(rem_ceps, "./tsne_result_basis.npy")
            _ = func.plot_clusta("./tsne_result_basis.npy", numClusta=30)
            clusta_leader = func.plot_clusta("./tsne_result_basis.npy", numClusta=30)
            for i in range(len(clusta_leader)):
                np.save('./rem_for_basis_30_'+str(i), rem_stft[:,clusta_leader[i]])
            """
            stft = func.stft(train)
            ceps = func.extract_ceps(stft)
            a,b,c = func.calculate_clusta("./tsne_result_.npy")
            rem_stft = func.remove_noise(stft, a,b)
            rem_ceps = func.extract_ceps(rem_stft)
            init_F = []
            for i in range(30):
                init_F.append(np.load('./npys/rem_for_basis_30_'+str(i)+".npy"))
            init_F = np.array(init_F)
            # F.shape=(30, 1025)
            init_G = np.random.rand(init_F.T.shape[1], rem_stft.shape[1])
            print("aaaaa",init_G)
            F, G, cost = func.train_vvvf(rem_stft, init_F=init_F.T, init_G=init_G, init='custom', beta=0)
            target_stft = func.stft(target)
            ssnmf = func.extract_vvvf(target_stft, F=F, beta=0, p=True)

            """remした後にplot
            rem_nmf = func.train_vvvf(rem_stft)
            ssnmf = func.extract_vvvf(rem_nmf, target)
            func.plot_amptitudemap(ssnmf)
            #func.plot_nmf_harmonic_structure(rem_nmf)
            #rem_cps = func.extract_ceps(rem_nmf[0])
            #func.plot_nmf_harmonic_structure(rem_cps.T)
            #func.save_tsne_result(rem_cps, fname)
            #clusta_leader = func.plot_clusta(fname)
            #func.plot_clusta_leader(rem_nmf[0].T, rem_cps, clusta_leader)
            """

    return ssnmf#, rem_cps, clusta_leader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '教師データからNMFを使って特徴を抽出し，\
                                                    雑音重畳音源から同様の音を抽出するプログラム')
     # 教師データ用のファイルリスト
    train_flist = ["vvvf_seibu6000","vvvf_eidan7000","vvvf_tobu9050","vvvf_tobu50000","vvvf_tokyu5000"]
     # 対象の雑音重畳音源のファイルリスト
    target_flist = ["train_seibu6000_1","train_seibu6000_2", "train_seibu6000_3"]

    parser.add_argument('--train', default=train_flist, choices=train_flist,
                         nargs='*', help="教師データファイルを選択")
    parser.add_argument('--target', default=target_flist, choices=target_flist,
                         nargs='*', help='解析したいデータファイルを選択')
    parser.add_argument('--basis', default=15, type=int, help='対象のNMFの規定数を指定')    
    args = parser.parse_args()

    a = main(args)
