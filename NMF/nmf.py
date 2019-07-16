import numpy as np
from sklearn.decomposition import NMF as nmf
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

def NMF(Y, R=3, n_iter=50, init_F=None, init_G=None, init='random', beta=2, verbose=False):
    loss_key = {0 : 'itakura-saito', 1 : 'kullback-leibler', 2 : 'frobenius'}
    if beta is 2:
        model = nmf(n_components=R, init=init, random_state=0, beta_loss=loss_key[beta])
    else :
        model = nmf(n_components=R, init=init, random_state=0, beta_loss=loss_key[beta], solver='mu',)

    if init is 'random':
        F = model.fit_transform(Y)
        G = model.components_
    elif init is 'custom':
        init_F = np.ascontiguousarray(init_F, dtype='double')
        init_G = np.ascontiguousarray(init_G, dtype='double')
        Y = np.ascontiguousarray(Y, dtype='double')
        F = model.fit_transform(Y, W=init_F, H=init_G)
        G = model.components_

    Lambda = np.dot(F, G)
    if beta == 0:
        cost = IS_divergence(Y, Lambda)
    elif beta == 1:
        cost = KL_divergence(Y, Lambda)
    else :
        cost = euclid_divergence(Y, Lambda)
    print("NMF extract")

    return [F, G, cost]

def SSNMF(Y, R=3, n_iter=50, F=[], init_G=[], init_H=[], init_U=[], beta=2, p =None, verbose=False):
    """
    decompose non-negative matrix to components and activation with Semi-Supervised NMF
    
    Y ≈　FG + HU
    Y ∈ R (m, n)
    F ∈ R (m, x)
    G ∈ R (x, n)
    H ∈ R (m, k)
    U ∈ R (k, n)

    parameters
    ---- 
    Y: target matrix to decompose
    R: number of bases to decompose
    n_iter: number for executing objective function to optimize
    F: matrix as supervised base components
    init_W: initial value of W matrix. default value is random matrix
    init_H: initial value of W matrix. default value is random matrix

    return
    ----
    Array of:
    0: matrix of F
    1: matrix of G
    2: matrix of H
    3: matrix of U
    4: array of cost transition
    
    ----
    beta :
    0: Itakura-Saito distance
    1: KL divergence
    2: Frobenius norm
    """
    
    eps = np.spacing(1)

    # size of input spectrogram
    M = Y.shape[0];
    N = Y.shape[1]; 
    X = F.shape[1]
    
    # initialization
    if len(init_G):
        G = init_G
        X = init_G.shape[1]
    else:
        G = np.random.rand(X, N)
        
    if len(init_U):
        U = init_U
        R = init_U.shape[0]
    else:
        U = np.random.rand(R, N)

    if len(init_H):
        H = init_H;
        R = init_H.shape[1]
    else:
        H = np.random.rand(M, R)

    # array to save the value of the euclid divergence
    cost = np.zeros(n_iter)
    # computation of Lambda (estimate of Y)
    Lambda = np.dot(F, G) + np.dot(H, U)
    # Set the basis punitive factor
    pena = 1.0 * 10**6
    # beta = 0 (Itakura-Saito) or beta = 1 (KL-divergence) or beta = 2 (Frobenius)
    # IS-divergence
    if beta == 0:
    # iterative computation
        for it in range(n_iter):
            H_copy = H.copy()
            U_copy = U.copy()
            G_copy = G.copy()
            HU_FG = np.dot(H, U) + np.dot(F, G) + eps
            print("SSNMF extract : {}".format(it))
            if p == None:
                for h_w in range(H.shape[0]):
                    for h_i in range(H.shape[1]):
                        H[h_w][h_i] = H_copy[h_w][h_i] * (sigma_3matrix(np.power(HU_FG[h_w], -2), Y[h_w], U_copy[h_i]) + eps) / sigma_2matrix(np.power(HU_FG[h_w], -1), U_copy[h_i])
            else :
                for h_w in range(H.shape[0]):
                    for h_i in range(H.shape[1]):
                        H[h_w][h_i] = H_copy[h_w][h_i] * (sigma_3matrix(np.power(HU_FG[h_w], -2), Y[h_w], U_copy[h_i]) + eps) / (sigma_2matrix(np.power(HU_FG[h_w], -1), U_copy[h_i]) + pena * H_copy[h_w][h_i] * np.power(F[h_w], 2).sum() + eps)

            for u_i in range(U.shape[0]):
                for u_t in range(U.shape[1]):
                    U[u_i][u_t] = U_copy[u_i][u_t] * (sigma_3matrix(np.power(HU_FG[:,u_t], -2), Y[:,u_t], H_copy[:,u_i]) + eps) / sigma_2matrix(np.power(HU_FG[:,u_t], -1), H_copy[:,u_i])

            for g_j in range(G.shape[0]):
                for g_t in range(G.shape[1]):
                    G[g_j][g_t] = G_copy[g_j][g_t] * (sigma_3matrix(np.power(HU_FG[:,g_t], -2), Y[:,g_t], F[:,g_j]) + eps) / sigma_2matrix(np.power(HU_FG[:,g_t], -1), F[:,g_j])

            # compute IS divergence
            cost[it] = IS_divergence(Y, Lambda + eps)

            # recomputation of Lambda (estimate of V)
            Lambda = np.dot(H, U) + np.dot(F, G)
            
        return [F, G, H, U, cost]

    # KL-divergence
    if beta == 1:
    # iterative computation
        for it in range(n_iter):
            H_copy = H.copy()
            U_copy = U.copy()
            G_copy = G.copy()
            HU_FG = np.dot(H, U) + np.dot(F, G) + eps
            print("SSNMF extract : {}".format(it))
            
            if p == None:
                for h_w in range(H.shape[0]):
                    for h_i in range(H.shape[1]):
                        H[h_w][h_i] = H_copy[h_w][h_i] * (sigma_3matrix(np.power(HU_FG[h_w], -1), Y[h_w], U_copy[h_i]) + eps) / (U_copy[h_i].sum() + eps)
            else :
                for h_w in range(H.shape[0]):
                    for h_i in range(H.shape[1]):
                        H[h_w][h_i] = H_copy[h_w][h_i] * (sigma_3matrix(np.power(HU_FG[h_w], -1), Y[h_w], U_copy[h_i]) + eps) / (U_copy[h_i].sum() + pena * H_copy[h_w][h_i] * np.power(F[h_w], 2).sum() + eps)

            for u_i in range(U.shape[0]):
                for u_t in range(U.shape[1]):
                    U[u_i][u_t] = U_copy[u_i][u_t] * (sigma_3matrix(np.power(HU_FG[:,u_t], -1), Y[:,u_t], H_copy[:,u_i]) + eps) / (H_copy[:,u_i].sum() + eps)

            for g_j in range(G.shape[0]):
                for g_t in range(G.shape[1]):
                    G[g_j][g_t] = G_copy[g_j][g_t] * (sigma_3matrix(np.power(HU_FG[:,g_t], -1), Y[:,g_t], F[:,g_j]) + eps) / (F[:,g_j].sum() + eps)

            # compute KL divergence
            cost[it] = KL_divergence(Y, Lambda + eps)

            # recomputation of Lambda (estimate of V)
            Lambda = np.dot(H, U) + np.dot(F, G)
            
        return [F, G, H, U, cost] 

    if beta == 2:
    # iterative computation
        for it in range(n_iter):
            print("SSNMF extract : {}".format(it))

            # compute euclid divergence
            cost[it] = euclid_divergence(Y, Lambda + eps)
            
            # will change
            if p == None:
                # update of H
                H *= (np.dot(Y, U.T) + eps) / (np.dot(np.dot(H, U) + np.dot(F, G), U.T) + eps)
            else :
                H *= (np.dot(Y, U.T) + eps) / (np.dot(np.dot(H, U) + np.dot(F, G), U.T) + eps)

            # update of U
            U *= (np.dot(H.T, Y) + eps) / (np.dot(H.T, np.dot(H, U) + np.dot(F, G)) + eps)
            
            # update of G
            G *= (np.dot(F.T, Y) + eps)[np.arange(G.shape[0])] / (np.dot(F.T, np.dot(H, U) + np.dot(F, G)) + eps)
            
            # recomputation of Lambda (estimate of V)
            Lambda = np.dot(H, U) + np.dot(F, G)
            
        return [F, G, H, U, cost]

#間違ってるかも・・・
def IS_divergence(Y, Yh):
    d = np.sum(np.abs(np.where(Y != 0, (Y / Yh) - np.log2(Y / Yh) - 1, 0)))
    return d

#間違ってるかも・・・
def KL_divergence(Y, Yh):
    d = np.sum(np.abs(np.where(Y != 0, Y * np.log2(Y / Yh), 0)))
    return d

def euclid_divergence(Y, Yh):
    d = 1 / 2 * (Y ** 2 + Yh ** 2 - 2 * Y * Yh).sum()
    return d

def sigma_2matrix(P, S):
    sum_ = 0
    for i in range(len(P)):
        sum_ += P[i] * S[i]
    return sum_

def sigma_3matrix(P, S, U):
    sum_ = 0
    for i in range(len(P)):
        sum_ += P[i] * S[i] * U[i]
    return sum_



#実装のメモ (行列の積で表現しようとしてうまくいかんやったやつ)
    """行列の積でうまくいかんやったやつ
    K = G
    J = G
    D = G
    K = G
    J = G
    D = G
    if beta == 0:
    # iterative computation
        for it in range(n_iter):
            print(H)
            print(U)
            print("HHHHHHHHHHHHHUUUUUUUUUUUUUU")
            print(F)
            print(G)
            print("FFFFFFFFFFFFFGGGGGGGGGGGGGG")
            print(np.dot(H, U) + np.dot(F, G))
            print("HU+FG!!!!!!!!!!!!!!!!!!!!!!!")
            print(D)
            print("DDDDDDDDDDDDDDDDDDDDDDDDDDD")
            print(Lambda)
            print("YYYYYYYYYYYYYYYYYYYYYYYYYYY")

            # compute IS divergence
            #cost[it] = IS_divergence(Y, Lambda + eps)
            cost[it] = euclid_divergence(Y, Lambda + eps)
            print("Y:{},F:{},G:{},H:{},U:{}".format(Y.shape,F.shape,G.shape,H.shape,U.shape))

            pow_ = np.power(np.dot(H, U) + np.dot(F, G) + eps , -2)
            pow_[np.isnan(pow_)] = 0

            # update of H           
            H *= (np.dot(pow_ * Y, U.T) + eps) / (np.dot(np.power(np.dot(H, U) + np.dot(F, G), -1), U.T) + eps)

            # update of U
            U *= (np.dot(H.T, pow_ * Y) + eps) / (np.dot(H.T, np.power(np.dot(H, U) + np.dot(F, G) + eps, -1)) + eps)
            
            # update of G
            G *= (np.dot(F.T, pow_ * Y) + eps) / (np.dot(F.T, np.power(np.dot(H, U) + np.dot(F, G) + eps, -1)) + eps)

            K *= (np.dot(F.T, np.power(np.dot(H, U) + np.dot(F, G) + eps, -2) * Y) + eps)
            D = np.power(np.dot(H, U) + np.dot(F, G) + eps, -2)
            J *= (np.dot(F.T, np.power(np.dot(H, U) + np.dot(F, G), -1)) + eps)

            # recomputation of Lambda (estimate of V)
            Lambda = np.dot(H, U) + np.dot(F, G)
            
        return [F, G, H, U, cost]
    """
    """
   # np.powをするのかどうか・・・前実装通りだとやってない，無視してる
    if beta == 1:
    # iterative computation
        for it in range(n_iter):

            # compute KL divergence
            #cost[it] = KL_divergence(Y, Lambda + eps)
            cost[it] = euclid_divergence(Y, Lambda + eps)

            # update of H
            H *= (np.dot(np.power(np.dot(H, U) + np.dot(F, G), -1) * Y.T, U.T) + eps) / ((np.dot(U.T, np.power(np.dot(H, U) + np.dot(F, G), -1))) + eps)

            # update of U
            U *= (np.dot(H.T, np.power(np.dot(H, U) + np.dot(F, G), -1) * Y.T).T + eps) / ((np.dot(np.power(np.dot(H, U) + np.dot(F, G), 0).T), U) + eps)

            # update of G
            G *= (np.dot(F.T, np.power(np.dot(H, U) + np.dot(F, G), -1) * Y.T).T + eps) / ((np.dot(np.power(np.dot(H, U) + np.dot(F, G), 0).T), U) + eps)

            # recomputation of Lambda (estimate of V)
            Lambda = np.dot(H, U) + np.dot(F, G)
            
        return [F, G, H, U, cost]
    """

#NMFの実装，実装後にライブラリを見つけたので，ライブラリを使用
    """
    def NMF(Y, R=3, n_iter=50, init_H=[], init_U=[], verbose=False):

    decompose non-negative matrix to components and activation with NMF
    
    Y ≈　HU
    Y ∈ R (m, n)
    H ∈ R (m, k)
    HU ∈ R (k, n)
    
    parameters
    ---- 
    Y: target matrix to decompose
    R: number of bases to decompose
    n_iter: number for executing objective function to optimize
    init_H: initial value of H matrix. default value is random matrix
    init_U: initial value of U matrix. default value is random matrix
    
    return
    ----
    Array of:
    0: matrix of H
    1: matrix of U
    2: array of cost transition

    eps = np.spacing(1)

    # size of input spectrogram
    M = Y.shape[0]
    N = Y.shape[1]
    
    # initialization
    if len(init_U):
        U = init_U
        R = init_U.shape[0]
    else:
        U = np.random.rand(R,N);

    if len(init_H):
        H = init_H;
        R = init_H.shape[1]
    else:
        H = np.random.rand(M,R)

    # array to save the value of the euclid divergence
    cost = np.zeros(n_iter)

    # computation of Lambda (estimate of Y)
    Lambda = np.dot(H, U)

    # iterative computation
    for i in range(n_iter):

        # compute euclid divergence
        cost[i] = euclid_divergence(Y, Lambda)

        # update H
        H *= np.dot(Y, U.T) / (np.dot(np.dot(H, U), U.T) + eps)
        
        # update U
        U *= np.dot(H.T, Y) / (np.dot(np.dot(H.T, H), U) + eps)
        
        # recomputation of Lambda
        Lambda = np.dot(H, U)

    return [H, U, cost]
    """