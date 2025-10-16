# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/house_prices.csv")

print("📄 Dataset preview:")
print(df.head())

# -----------------------------
# 2. Split Features and Target
# -----------------------------
X = df.drop("price", axis=1)
y = df["price"]

print(X)
print(y)

# -----------------------------
# 3. Preprocessing
# -----------------------------
numeric_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
categorical_features = ["city"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 4. Build Pipeline
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# -----------------------------
# 5. Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 6. Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

print("\n📈 Model Evaluation:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# -----------------------------
# 8. Test with New Data
# -----------------------------
sample = pd.DataFrame({
    "area": [3800],
    "bedrooms": [3],
    "bathrooms": [2],
    "stories": [2],
    "parking": [1],
    "city": ["Lahore"]
})

predicted_price = model.predict(sample)
print(f"\n💰 Predicted Price for {sample.iloc[0].to_dict()} = Rs. {predicted_price[0]:,.0f}")
