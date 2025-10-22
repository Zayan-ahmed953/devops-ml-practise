import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("data/data.csv")

normalize = StandardScaler()
X_normalize = normalize.fit_transform(df)

pca = PCA(n_components=2)
Xpca = pca.fit_transform(X_normalize)

model = DBSCAN(eps=1.2, min_samples=3)
label = model.fit_predict(Xpca)

plt.scatter(Xpca[:,0], Xpca[:,1], c=label, s=100, cmap='viridis', edgecolors="k")
plt.title("yoyuo")
plt.xlabel("pc1 component 1")
plt.ylabel("pc1 component 2")
plt.show()

n_cluster = len(set(label)) - (1 if -1 in label else 0)
noise = list(label).count(-1)

if n_cluster > 1:
    sc = silhouette_score(Xpca, label)
    print(sc)
else:
    print('Not enough clusters')