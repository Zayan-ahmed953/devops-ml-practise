import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


df = pd.read_csv("data/data.csv")

X = df[['Amount', 'Location_Score']]

wscc = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wscc.append(kmeans.inertia_)

plt.plot(range(1, 11), wscc, marker='o')
plt.title('Anomaly detection')
plt.xlabel('Clusters')
plt.ylabel('WSCC')
plt.show()


model = KMeans(n_clusters=4, random_state=42)
df['Clusters'] = model.fit_predict(X)

print(df)

plt.figure(figsize=(7,5))
for cluster_id in sorted(df['Clusters'].unique()):
    subset = df[df['Clusters'] == cluster_id]
    plt.scatter(subset['Amount'], subset['Location_Score'], label=f'Cluster {cluster_id}')

plt.title('CLustering')
plt.xlabel('Amount for transaction')
plt.ylabel('Location in Score')
plt.legend(title='Clusters')
plt.show()

labels = df['Clusters']

selhite_score = silhouette_score(X, labels)

print(selhite_score)