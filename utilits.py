import os 

import numpy as np
from numpy.lib.type_check import imag
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

import config

def getAllName(dir):
    fileList = list(map(lambda fileName: os.path.join(dir, fileName), os.listdir(dir)))
    return fileList

#  Clustering functions
def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * K + 1
    # print(C.shape)
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L

def spectral_clustering(C, K, d, alpha, ro):
    C = thrC(C, alpha)
    y, _ = post_proC(C, K, d, ro)
    return y

# Reconstruction functions
def reconLabel(labels, pixelBlockList):
    '''
    input_param:
        labels: Subspace clustering labels.
        pixelBlockList: A list contains all element.
    output_param:
        reconLabel: reconstructed label matrix.
    '''
    
    reconLabel = np.zeros(config.imgSize)
    for label, pixelBlock in zip(labels, pixelBlockList):
        
        pixelBlock = pixelBlock.reshape((-1,3))
        pixelBlock = list(pixelBlock)
        #print(pixelBlock.shape)
        for idx, item in enumerate(pixelBlock):
            # print(item)
            if (item == config.blankBlock).all():
                pixelBlock[idx] = 0
                
            else:
                pixelBlock[idx] = label

        pixelBlock = np.array(pixelBlock)
        pixelBlock = pixelBlock.reshape(config.imgSize)
        reconLabel = reconLabel + pixelBlock

    return reconLabel

if __name__ == '__main__':
    test =[[[1, 1, 1],[0, 0, 0],[0, 0, 0]],
                    [[0, 0, 0],
                    [2, 2, 2],
                    [0, 0, 0]],
                    [[0, 0, 0],
                    [0, 0, 0],
                    [3, 3, 3]]]

    test = np.array(test)
    print(test.shape)

    labels = [1 ,0 , 1]

    recon = reconLabel(labels, test)
    print(recon)