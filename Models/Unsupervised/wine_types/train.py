import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
plt.ylabel('Sugar')


df = pd.read_csv('data/data.csv')

X = df[['Volatile_Acidity', 'Residual_Sugar']]

wcss = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, marker = 'o')
plt.title('Elbow method')
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')

plt.show()

model = KMeans(n_clusters=4, random_state=42)
df['Clusters'] = model.fit_predict(X)

print(df)

plt.scatter(df['Volatile_Acidity'], df['Residual_Sugar'], c = df['Clusters'])
plt.title('THis is the predicted')
plt.xlabel('Acidity')
plt.ylabel('Sugar')
plt.show()


