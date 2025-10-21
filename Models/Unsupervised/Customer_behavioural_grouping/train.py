import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv("data/data.csv")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

model = DBSCAN(eps=2, min_samples=4)
labels = model.fit_predict(X_pca)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=100, edgecolors='k')
plt.title('DBSCAN with dimensionality reduction')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.show()