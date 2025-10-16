import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score





df = pd.read_csv("data/data.csv")
df = pd.DataFrame(df)

x = df.drop('Price', axis=1)
y = df['Price']

categrocial = ['Location']
numerical = ['AreaSqFt' , 'Bedrooms' , 'Bathrooms' , 'Age' , 'HasGarage' , 'NearbySchoolRating']

preproccesor = ColumnTransformer (
    transformers= [
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(), categrocial)
    ]
)

ridge = Ridge(alpha=1.0)

model = Pipeline ([
    ('preproccesor', preproccesor),
    ('regressor', ridge)
])

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

mean = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print('Avergae error is: ',mean)
print('\n\nR2 Score is: ',r2)