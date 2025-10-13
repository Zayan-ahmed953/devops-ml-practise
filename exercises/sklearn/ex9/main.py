from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

pipe.fit(X_train, Y_train)

predict = pipe.predict(X_test)

accuracy = accuracy_score(Y_test, predict)

print(accuracy)

# TODO:
# 1. Create a pipeline with ('scaler', StandardScaler()) and ('svc', SVC())
# 2. Fit and evaluate on train/test data

