import numpy as np
import sys
import pandas as pd
from KMeans import K_means


def Calinski_Harabasz_score(X,labels):
    n_samp = len(X)
    K = max(labels)+1
    if K== 1:
        return sys.maxsize

    group_disp = 0
    within_disp = 0
    mean_all = np.mean(X,axis=0)

    for k in range(K):
        C_k = X[labels==k]
        m_k = np.mean(C_k,axis=0)
        group_disp+= np.sum(np.square(m_k-mean_all))*len(C_k)
        within_disp +=np.sum(np.square(C_k-m_k))

    return group_disp * (n_samp-K)/(within_disp*(K-1))

if __name__ == '__main__':
    df = pd.read_excel(r'data.xlsx')
    X = df.to_numpy()
    X = X[:,:3]

    for i in range(2,10):
        labels,centroid = K_means(X,i,50)
        print('Calinski-Harabasz Index for k=',i,' ',Calinski_Harabasz_score(X,labels))