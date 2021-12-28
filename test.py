import pandas as pd
from sklearn import cluster
from  adjusted_rand_index import ARI
import matplotlib.pyplot as plt
from sklearn import metrics
from KMeans import Feature_normalize
import numpy as np

df = pd.read_excel(r'data.xlsx')
#X = df.to_numpy()[:,0]
#X=Feature_normalize(X)
#obj = cluster.AgglomerativeClustering(n_clusters=2).fit(X)
#print(obj.labels_)
print(df.head())
