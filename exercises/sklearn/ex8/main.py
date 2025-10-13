from sklearn.model_selection import GridSearchCV
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

iris = load_iris()

x = iris.data
y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

dct = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dct.fit(X_train_scaled, Y_train)

predict = dct.predict(X_test_scaled)

accurany_dct = accuracy_score(Y_test, predict)

print(accurany_dct)

# TODO:
# 1. Set up a param_grid for DecisionTree (e.g., max_depth, min_samples_split)
# 2. Use GridSearchCV to find the best params
# 3. Print best parameters and best score


