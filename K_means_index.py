import numpy as np
import sys
import pandas as pd
from KMeans import K_means
from Silhouette_Score import Silhouette_score
from Calinski_Harbaz import Calinski_Harabasz_score
from adjusted_rand_index import ARI
import matplotlib.pyplot as plt


def open_file(file_name):
    # open file type
    #df = pd.read_excel(file_name)
    df = pd.read_csv(file_name)
    X = df.to_numpy()
    return X


def plot_hist(x_axis,S_score_array,cal_h_array):
    # function yo plot histogram
    fig = plt.figure()
    fig.suptitle('K-Means Clustering Algorithm', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(211)
    bx = fig.add_subplot(212)
    ax.bar(x_axis, S_score_array, color='r', width=0.25)
    ax.set_title("Silhouette score")

    bx.bar(x_axis, cal_h_array, color='b', width=0.25)
    bx.set_title("Calinski Harabaz score")
    bx.set_xlabel('Number of clusters')
    plt.show()


def KMeans_Algo(file_name,start,end):
    X = open_file(file_name)
    print('Shape of data = ', X.shape)
    m_sc = -1000
    K_sc = -1
    m_cal_h = -1000
    K_cal_h = -1

    x_axis = np.arange(start, end)
    S_score_array = []
    cal_h_array = []

    for i in range(start, end):
        labels, centroid = K_means(X, i, 100)
        # print(labels)
        S_c = Silhouette_score(X, labels)
        cal_h = Calinski_Harabasz_score(X, labels)
        S_score_array.append(S_c)
        cal_h_array.append(cal_h)

        if m_sc < S_c:
            m_sc = S_c
            K_sc = i
        if m_cal_h < cal_h:
            m_cal_h = cal_h
            K_cal_h = i

        # print('Silhouette score for   K =', i, ' ', S_c)
        # print('Calinski Harabaz score K =', i, ' ',cal_h,'\n' )
        # print('ARI: ',ARI(labels,X[:,-1]))
    plot_hist(x_axis,S_score_array,cal_h_array)
    print('Max Silhouette score is       ', m_sc, 'for K =', K_sc)
    print('Max Calinski Harabaz score is ', m_cal_h, 'for K =', K_cal_h)

if __name__ == '__main__':
    KMeans_Algo('sobar-72.csv', 2, 12)







