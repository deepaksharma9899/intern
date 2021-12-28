import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Normalize import Feature_normalize
import sys
from Silhouette_Score import Silhouette_score
from Calinski_Harbaz import Calinski_Harabasz_score


def plot_hist(x_axis,S_score_array,cal_h_array):
    fig = plt.figure()
    fig.suptitle('Hierarchical Clustering Algorithm', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(211)
    bx = fig.add_subplot(212)
    ax.bar(x_axis, S_score_array, color='b', width=0.25)
    ax.set_title("Silhouette score")

    bx.bar(x_axis, cal_h_array, color='r', width=0.25)
    bx.set_title("Calinski Harabaz score")
    bx.set_xlabel('Number of clusters')
    plt.show()


def proximity_matrix(X):
    prox_mat = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            prox_mat[i][j] = np.sqrt(sum(np.square(np.subtract(X[i], X[j]))))

    return prox_mat

#Agglomerative Clustering Algorithm
def HCA(labels,X,prox_mat):
    dis = sys.maxsize
    ind1=0
    ind2=0
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            temp=0
            for l in labels[i]:
                for m in labels[j]:
                    temp+=prox_mat[l][m]
        #temp is used for calculating average linkage
            temp/=(len(labels[i])*len(labels[j]))
            if dis>temp:
                dis=temp
                ind1=i
                ind2=j
    labels2=[]

    for i in range(len(labels)):
        if i!=ind1 and i!=ind2:
            labels2.append(labels[i])

    t=[]
    for i in labels[ind1]:
        t.append(i)
    for i in labels[ind2]:
        t.append(i)

    labels2.append(t)

    return labels2


def Hierarchical_label(X,K):
    # function to find labels at a given K

    # labels stores the indexes of cluster formed
    labels = np.arange(0, len(X), dtype=int).reshape(-1, len(X)).T

    # prox_mat stores the euclidian distances between different points
    prox_mat = proximity_matrix(X)

    # when no of cluster reduces to K our algorithm stops there
    for i in range(len(X)):
        labels = HCA(labels, X, prox_mat)
        if len(labels) == K:
            break

    label = np.zeros(len(X), dtype=int)
    for j in range(K):
        label[labels[j]] = j
    return label


def Hierarchical_Algo(file_name, start, end):
    #df = pd.read_excel(file_name)
    df = pd.read_csv(file_name)
    X = df.to_numpy()
    X = Feature_normalize(X)

    # labels stores the indexes of cluster formed
    labels = np.arange(0, len(X), dtype=int).reshape(-1, len(X)).T

    # prox_mat stores the euclidian distances between different points
    prox_mat = proximity_matrix(X)

    # when no of cluster reduces to K our algorithm stops there
    m_sc = -1000
    K_sc =-1
    m_cal_h = -1000
    K_cal_h = -1

    x_axis = []
    S_score_array = []
    cal_h_array = []

    for i in range(start,len(X)):
        labels = HCA(labels, X,prox_mat)
        K=len(labels)

        if K < end:
            x_axis.append(K)
            label = np.zeros(len(X),dtype=int)
            for j in range(K):
                label[labels[j]]=j
            S_c = Silhouette_score(X, label)
            cal_h = Calinski_Harabasz_score(X, label)
            S_score_array.append(S_c)
            cal_h_array.append(cal_h)

            if m_sc < S_c:
                m_sc = S_c
                K_sc = K
            if m_cal_h < cal_h:
                m_cal_h = cal_h
                K_cal_h = K

            #print('Silhouette score for   K =', K, ' ', S_c)
            #print('Calinski Harabaz score K =', K, ' ', cal_h, '\n')
    plot_hist( x_axis, S_score_array, cal_h_array)
    print('Max Silhouette score is       ', m_sc, 'for K =', K_sc)
    print('Max Calinski Harabaz score is ', m_cal_h, 'for K =', K_cal_h)


if __name__ == '__main__':
    Hierarchical_Algo('sobar-72.csv', 2, 10)
