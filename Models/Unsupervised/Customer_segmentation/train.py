import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("data/data.csv")

X = df[["Annual_Income", "Spending_Score"]]


wscc = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wscc.append(kmeans.inertia_)

plt.plot(range(1, 11), wscc, marker = "o")
plt.title("W")
plt.xlabel("xxx")
plt.ylabel("yyy")
plt.show()

model = KMeans(n_clusters=4)

df["Clusters"] = model.fit_predict(X)

print(df)



plt.scatter(df["Annual_Income"], df["Spending_Score"], c=df["Clusters"])

plt.show()

labels = df['Clusters']

selhite_scroe = silhouette_score(X, labels)

print(selhite_scroe)