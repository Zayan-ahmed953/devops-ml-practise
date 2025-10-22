import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('data/data.csv')

df_numeric = df.select_dtypes(include=[np.number])
print(df_numeric.head())

normalize = StandardScaler()
X_normalized = normalize.fit_transform(df_numeric)

pca = PCA(n_components=2)
xpca = pca.fit_transform(X_normalized)

model = DBSCAN(eps=1.2, min_samples=3)
labels = model.fit_predict(xpca)

plt.scatter(xpca[:,1], xpca[:,2], c=labels, cmap='viridis', s=100, edgecolors="k")
plt.title("yoyoyoyo")
plt.xlabel("X label")
plt.ylabel("Y label")
plt.show()

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("Number of clusters found:", n_clusters)
print("Number of noise points:", n_noise)

