from sklearn.feature_selection import SelectKBest, f_classif
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
    ('svm', SVC())
])

pipe.fit(X_train, Y_train)

predict = pipe.predict(X_test)

accuracy = accuracy_score(Y_test, predict)

print(round(accuracy,3))

pipe_kbest = Pipeline([
    ('select', SelectKBest(score_func=f_classif, k=2)),
    ('scalar', StandardScaler()),
    ('svc',SVC())
])

pipe_kbest.fit(X_train, Y_train)

predict_kbest = pipe_kbest.predict(X_test)

accuracy_kbest = accuracy_score(Y_test, predict_kbest)
print(round(accuracy_kbest,3))

# TODO:
# 1. Use SelectKBest to pick top 2 features
# 2. Train model using only selected features
# 3. Compare accuracy with full-feature model
