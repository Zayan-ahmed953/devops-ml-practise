import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_rows = 500

data = {
    'age': np.random.randint(18, 65, num_rows),
    'gender': np.random.randint(0, 2, num_rows),
    'income': np.random.randint(20000, 120000, num_rows),
    'married': np.random.randint(0, 2, num_rows),
    'education_level': np.random.randint(1, 5, num_rows),
    'num_purchases_last_month': np.random.randint(0, 15, num_rows),
    'visited_website': np.random.randint(0, 2, num_rows),
}

# Define a relationship for target variable
# Higher income + visited website + more purchases => more likely to buy
probability = (
    0.3 * (data['income'] / 120000) +
    0.2 * data['visited_website'] +
    0.3 * (data['num_purchases_last_month'] / 15) +
    0.1 * (data['education_level'] / 4) +
    0.1 * data['married']
)

# Convert probability into binary target
data['bought_product'] = (probability + np.random.normal(0, 0.1, num_rows)) > 0.5
data['bought_product'] = data['bought_product'].astype(int)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/customer_purchase_data.csv', index=False)

print(df)

x = df.drop('bought_product', axis = 1)
y = df['bought_product']

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=42)

pipe = Pipeline ([
    ('scaler', StandardScaler()),
    ('lgg', LogisticRegression())
])

pipe.fit(X_train, Y_train)

predict = pipe.predict(X_test)

accuracy = accuracy_score(Y_test, predict)


print(round(accuracy, 3))


# sample_user = {
#     'age': 19,
#     'gender': 1,
#     'income': 80000,
#     'married': 0,
#     'education_level': 3,
#     'num_purchases_last_month': 2,
#     'visited_website': 1,
# }

# sd = pd.DataFrame([sample_user])

# sg_predict = pipe.predict(sd)
# sg_proba = pipe.predict_proba(sd)


# print(sg_predict)
# print(sg_proba)