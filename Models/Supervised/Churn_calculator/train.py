import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



raw_data = pd.read_csv('data/data.csv')

rd = pd.DataFrame(raw_data)
# print(rd.head)

x = rd.drop('ChurnRisk', axis=1)
y = rd['ChurnRisk']

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=22)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rff', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipe.fit(X_train, Y_train)

predict = pipe.predict(X_test)

accuracy = accuracy_score(Y_test, predict)

# print(accuracy)

sample_data = {
  "CustomerTenure": 20,
  "MonthlyCharges": 80,
  "TotalCharges": 1600,
  "NumComplaints": 2,
  "CustomerSupportCalls": 3,
  "InternetUsageGB": 120,
  "ContractType": 1,
  "PaymentMethod": 3,
  "HasStreamingServices": 1
}

sd = pd.DataFrame([sample_data])

predict = pipe.predict(sd)

print(predict)