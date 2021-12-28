import numpy as np

def Feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    x_norm = np.subtract(X , mu)
    x_norm = np.divide(x_norm, sigma)

    return x_norm