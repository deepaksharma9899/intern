import numpy as np
import sys
import pandas as pd
from KMeans import K_means
# calcutes euclidean distance

def Euc_dist(x1,x2):
    return np.sqrt(np.sum(np.square(np.subtract(x1,x2))))

def dist_matrix(X):
    n=len(X)
    dist_mat = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            dist_mat[i,j]=Euc_dist(X[i],X[j])

    return dist_mat


def Silhouette_samp(X,label):
    n_samp = len(X)
    S = np.zeros(n_samp)

    K = np.max(label)+1

    dist_mat=dist_matrix(X)

    # calculates silhouette value for each datapoint

    Ci = np.zeros((K),dtype=int)
    for j in range(n_samp):
        if label[j]==-1:
            continue
        Ci[label[j]]+=1


    for i in range(n_samp):
        if label[j]==-1:
            continue
        if Ci[label[i]]==1:
            continue
        D = np.zeros((K),dtype=float)
        for j in range(n_samp):
            D[label[j]]+=dist_mat[i,j]
            #print(D[label[i]],' ',label[i])
        Ci[label[i]]-=1
        Ai_Bi = np.divide(D,Ci)
        Ci[label[i]] += 1
        Ai = Ai_Bi[label[i]]
        Ai_Bi[label[i]] = sys.maxsize
        Bi = np.amin(Ai_Bi)

        S[i]=(Bi-Ai)/max(Bi,Ai)

    return S

#this functions calcultes mean of silhouette values
def Silhouette_score(X,label):
    if np.max(label)==0:
        return 1
    score = np.mean(Silhouette_samp(X,label))

    return score


if __name__ == '__main__':
    df = pd.read_excel(r'data.xlsx')
    X = df.to_numpy()
    X= X[:50,:3]

    for i in range(4,5):

        labels,centroid = K_means(X,i,50)
        print('Silhouette score for K =',i,' ',Silhouette_score(X,labels))


