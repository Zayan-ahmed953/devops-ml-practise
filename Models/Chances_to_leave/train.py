import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



Data = pd.read_csv('data/data.csv')

data = pd.DataFrame(Data)

print(data.head())


x = data.drop('WillLeave', axis=1)
y = data['WillLeave']

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

pipe.fit(X_train, Y_train)

predict = pipe.predict(X_test)

accuracy = accuracy_score(Y_test, predict)
print(accuracy)