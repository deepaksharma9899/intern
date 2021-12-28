

import math
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Function to find the closest centroid to an training example
def findClosestCentroids(X, centroids):
    K = len(centroids)
    idx = np.zeros(len(X), dtype=np.int)


    for i in range(len(X)):
        min1 = 99999999
        for j in range(K):
            if np.sum(np.square(np.subtract(X[i],centroids[j]))) < min1:
                min1 = np.sum(np.square(np.subtract(X[i],centroids[j])))
                idx[i] = j

    return idx

#function to find the centroid of dataset belonging to same clusters
def computeCentroids(X, idx, K):
    m,n = np.shape(X)

    centroids = np.zeros((K, n), dtype=np.float32)
    num = np.zeros(K, dtype=np.int)

    for i in range(m):
        for j in range(n):
            centroids[idx[i],j] = centroids[idx[i],j]+X[i,j]
        num[idx[i]] = num[idx[i]] + 1

    for i in range(K):
        centroids[i]=np.divide(centroids[i],num[i])

    return centroids

# function to randomly initialising the centroids
def kMeansInitCentroids(X, K):
    m,n = np.shape(X)

    centroids = np.zeros((K, n),dtype=np.float32)

    centroids = X[np.random.randint(0, m, size=(K))]

    return centroids

# main function
if __name__ == '__main__':
    # dataset
    x = make_blobs(n_samples=500,
                         centers=8,
                         n_features=2,
                         cluster_std=1.6,
                         random_state=250)
    X = np.array(x[0])

    #plt.scatter(X[:,0],X[:,1])
    #plt.show()
    #K : number of centroids
    K=16
    itert=25
    m,n=np.shape(X)
    WSS=[]
    


    for j in range(1,K):
      centroids = np.zeros((j, n),dtype=np.float32)
      centroids = kMeansInitCentroids(X, j)
      idx = np.zeros(m,dtype=np.int)
      for i in range(itert):
          idx=findClosestCentroids(X, centroids)
          centroids = computeCentroids(X, idx, j)
      
      SSE=0
      for i in range(j):
        T=X[idx==i]
        SSE+=np.sum(np.square(np.subtract(T,centroids[i])))
      WSS.append(SSE)
    


    #Implementing elbow method to calculte the optimum number of clusters
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    ax.title.set_text('Elbow method')
    plt.plot(np.arange(1,K),WSS)
    plt.ylabel('Within-Cluster-Sum of Squared(WSS)')
    plt.xlabel('No.of clusters')
    plt.show()
    # print("clusters calculated by Algorithm\n")
    # print(idx)
    # print("original clusters")
    # print(x[1])


#Evaluation of clusters

#1.Internal measures
    #1. cluster cohesion

    

    #print("SSE = ",SSE)

    #2. Cluster Sepration
#     BSS=0
#     mean=0
#     for i in range(K):
#         mean+=centroids[i]

#     mean= np.divide(mean,K)

#     for i in range(K):
#         s=sum(idx==i)
#         BSS += sum(np.square(np.subtract(mean, centroids[i])))*s

#     print("BSS = ", BSS)

# #2.External measures
#     #1.purity

#     Y= x[1]

#     class_centroid=computeCentroids(X, Y, K)

#     sum1=0

#     for i in range(K):
#         idx2 = np.zeros(K, dtype=np.int)
#         L = idx[Y==i]
#         #print(L)
#         max3=0
#         for j in L:
#             idx2[j]+=1
#             if max3<idx2[j]:
#                 max3=idx2[j]

#         sum1+=max3


#     purity= sum1/m

#     print("purity = ",purity)

#     #2. rand index


#     plt.figure("my_program")
#     plt.scatter(X[idx == 0, 0], X[idx == 0, 1], s=50, color='red')
#     plt.scatter(X[idx == 1, 0], X[idx == 1, 1], s=50, color='blue')
#     plt.scatter(X[idx == 2, 0], X[idx == 2, 1], s=50, color='yellow')
#     plt.scatter(X[idx == 3, 0], X[idx == 3, 1], s=50, color='cyan')
#     plt.scatter(centroids[0, 0], centroids[0, 1], marker='*', s=200, color='black')
#     plt.scatter(centroids[1, 0], centroids[1, 1], marker='*', s=200, color='black')
#     plt.scatter(centroids[2, 0], centroids[2, 1], marker='*', s=200, color='black')
#     plt.scatter(centroids[3, 0], centroids[3, 1], marker='*', s=200, color='black')

#     plt.figure("original_data")
#     plt.scatter(X[x[1] == 0, 0], X[x[1] == 0, 1], s=50, color='yellow')
#     plt.scatter(X[x[1] == 1, 0], X[x[1] == 1, 1], s=50, color='red')
#     plt.scatter(X[x[1] == 2, 0], X[x[1] == 2, 1], s=50, color='blue')
#     plt.scatter(X[x[1] == 3, 0], X[x[1] == 3, 1], s=50, color='cyan')
#     plt.show()

















#choose the k for which WSS becomes first starts to diminish. In the plot of WSS-versus-k, this is visible as an elbow.In the figure below it is 9