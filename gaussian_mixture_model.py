import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def Feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    x_norm = np.subtract(X , mu)
    x_norm = np.divide(x_norm, sigma)

    return x_norm


# Evalaute gaussian density function
def gaussian(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T
    determinant = np.linalg.det(cov)
    # if determinant ==0:
    #     print('Covariance matrix is Singular matrix')
    #

    inverse = np.linalg.inv(cov)
    temp = np.dot(np.dot(diff.T, inverse), diff)

    return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * determinant ** 0.5) * np.exp(-0.5 * temp)).reshape(-1, 1)

# function for initialising clusters
def initialize_parameters(X, n_clusters):
    clusters = []
    idx = np.arange(X.shape[0])
    #first n examples are taken as mean
    #mu_k = X[0:n_clusters,:]
    mu_k = X[np.random.randint(0, X.shape[0], size=(n_clusters))]
    
    # Initialising theta
    for i in range(n_clusters):
        clusters.append({
            'pi_k': 1.0 / n_clusters,
            'mu_k': mu_k[i],
            'cov_k': np.identity(X.shape[1], dtype=np.float64)
        })
        
    return clusters

# function for evaluating gamma 
def expect(X, clusters):
    totals = np.zeros((X.shape[0], 1), dtype=np.float64)
    
    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']
        temp = gaussian(X, mu_k, cov_k)
        gamma_nk = (pi_k * temp).astype(np.float64)
        
        for i in range(X.shape[0]):
            totals[i] += gamma_nk[i]
        
        cluster['gamma_nk'] = gamma_nk
        cluster['totals'] = totals
        
    
    for cluster in clusters:
        cluster['gamma_nk'] /= cluster['totals']



# function for evaluting mean, covariance matrix and mu
def maximize(X, clusters):
    N = float(X.shape[0])
  
    for cluster in clusters:
        gamma_nk = cluster['gamma_nk']
        cov_k = np.zeros((X.shape[1], X.shape[1]))
        
        N_k = np.sum(gamma_nk, axis=0)
        
        pi_k = N_k / N
        mu_k = np.sum(gamma_nk * X, axis=0) / N_k
        
        for j in range(X.shape[0]):
            diff = (X[j] - mu_k).reshape(-1, 1)
            cov_k += gamma_nk[j] * np.dot(diff, diff.T)
            
        cov_k /= N_k
        
        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k

# main function which is converging all the parameters 
def traindata(X, n_clusters, iter):
    X = Feature_normalize(X)
    clusters = initialize_parameters(X, n_clusters)
    scores = np.zeros((X.shape[0], n_clusters))

    for i in range(iter):
        expect(X, clusters)
        maximize(X, clusters)
        
    for i, cluster in enumerate(clusters):
        scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)

    label = np.argmax(scores,axis=1)
    return clusters,label

if __name__ == '__main__':
#n_clusters: No of clusters
# iter: Change no of iterations
#importing data
    df = pd.read_excel(r'data.xlsx')
    X = df.to_numpy()

    #X=X[:,:10]
    n_clusters = 8
    iter = 50

    clusters,scores= traindata(X, n_clusters, iter)
    print(scores)

    colorset = ['b', 'r', 'k','y','c','m','g']
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.title.set_text('Gaussian_mixture_model')

    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], c=colorset[scores[i]], marker='o')
    plt.show()
