import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



# Read data
df = pd.read_csv("data/data.csv")

X = df[["Butter", "Bread"]]


wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Relationship between Butter and Bread")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

model = KMeans(n_clusters=4, random_state=42)

df["Cluster"] = model.fit_predict(X)

# Get cluster centers
centers = model.cluster_centers_

# Plot clusters with legend
plt.figure(figsize=(7,5))
scatter = plt.scatter(df["Butter"], df["Bread"], 
                      c=df["Cluster"], 
                      cmap='viridis', 
                      s=100, alpha=0.8, edgecolors='k')

# Plot cluster centers
plt.scatter(centers[:, 0], centers[:, 1], 
            c='red', marker='o', s=200, label='Centroids')

# Create legend mapping cluster numbers to colors
legend_labels = [f"Cluster {i}" for i in range(model.n_clusters)]
legend = plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Clusters")

plt.scatter(df["Butter"], df["Bread"], c = df["Cluster"] )
plt.show()

labels = df["Cluster"]

silhotee = silhouette_score(X, labels)
print(silhotee)