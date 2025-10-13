from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

x = iris.data
y = iris.target

# TODO:
# 1. Train LogisticRegression, DecisionTree, and SVC
# 2. Compare their accuracies on test data
# 3. Print which performed best

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=42)

scaler=StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lgg = LogisticRegression()
lgg.fit(X_train_scaled, Y_train)

predicted_lgg = lgg.predict(X_test_scaled)
accuracy_lgg = accuracy_score(Y_test, predicted_lgg)

print("LogisticRegression accurancy:",accuracy_lgg)


dct = DecisionTreeClassifier()
dct.fit(X_train_scaled, Y_train)

precited_dct = dct.predict(X_test_scaled)
accuracy_dct = accuracy_score(Y_test, precited_dct)

print("\n\nDecisionTreeClassifier accurancy:",accuracy_dct)

svc = SVC()
svc.fit(X_train_scaled, Y_train)

predicted_svc = svc.predict(X_test_scaled)
accuracy_svc = accuracy_score(Y_test, predicted_svc)

print("\n\nSVC accurancy:",accuracy_dct)
