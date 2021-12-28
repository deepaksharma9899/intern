import numpy as np
import sys
import pandas as pd
from KMeans import K_means
from HCA_index import Hierarchical_label
from adjusted_rand_index import ARI
from fowlkes_mallows import Fowlkes_mallows
from gaussian_mixture_model import traindata


if __name__ == '__main__':
    file_name = 'sobar-72.csv'
    #df = pd.read_excel(file_name)
    df = pd.read_csv(file_name)
    X = df.to_numpy()
    label1, centroid = K_means(X, 2, 100)
    #label2 = Hierarchical_label(X,2)
    _,label3= traindata(X,2, 100)

    # print('K-means and Hierarchical clustering | file name :',file_name)
    # print('Adjusted Rand Index: ',ARI(label1,label2))
    # print('Fowlkes mallows score: ', Fowlkes_mallows(label1, label2))

    print('K-means and gaussian mixture model | file name :', file_name)
    print('Adjusted Rand Index: ', ARI(label1, label3))
    print('Fowlkes mallows score: ', Fowlkes_mallows(label1, label3))

    #print(label2)



