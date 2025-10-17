import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


df = pd.read_csv('data/data.csv')

X = df[['Acceleration_X', 'Heart_Rate']]

wscc = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wscc.append(kmeans.inertia_)

plt.plot(range(1,11), wscc, marker='o')

plt.title("WSCC analysis")
plt.xlabel("Acceleration_X")
plt.ylabel("Heart_Rate")
plt.show()


model = KMeans(n_clusters=4, random_state=42)

df['Clusters'] = model.fit_predict(X)
print(df)


plt.scatter(df['Acceleration_X'], df['Heart_Rate'], c=df['Clusters'])
plt.show()
