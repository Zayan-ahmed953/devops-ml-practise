import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv('data/data.csv')

X = df[['Annual_Income', 'Spending_Score']]

wcss = []  # Within Cluster Sum of Squares
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

Final_model = KMeans(n_clusters=4, random_state=42)

df['Clusters'] = Final_model.fit_predict(X)

plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Clusters'], camp = 'viridis')

plt.xlabel('Annual income (k$)')
plt.ylabel('Spending behaviour')
plt.title('Clustering customers by spending behaviour')
plt.show()