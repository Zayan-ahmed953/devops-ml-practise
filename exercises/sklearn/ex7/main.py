from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# TODO:
# 1. Train RandomForestClassifier
# 2. Use cross_val_score with cv=5
# 3. Print mean accuracy

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 ,random_state=42)

scale = StandardScaler()
scale.fit(x_train)

x_train_scaled = scale.transform(x_train)   
x_test_scaled = scale.transform(x_test)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(x_train_scaled, y_train)

predict = rfc.predict(x_test_scaled)

accuracy = accuracy_score(y_test, predict)

print(accuracy)

cv_scores = cross_val_score(rfc, x_train_scaled, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

