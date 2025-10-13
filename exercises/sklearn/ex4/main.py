from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



iris = load_iris()

x = iris.data
y = iris.target

# TODO:
# 1. Initialize KNeighborsClassifier(n_neighbors=3)
# 2. Fit on X_train, y_train
# 3. Predict on X_test
# 4. Print accuracy score


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print("Accuracy score", accuracy_score(y_test, y_pred))