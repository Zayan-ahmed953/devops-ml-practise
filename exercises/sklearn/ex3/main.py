from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

# Load the iris dataset
iris = datasets.load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# TODO:
# 1. Apply StandardScaler on X_train
# 2. Transform X_test
# 3. Print mean and std of scaled features

x = df.drop("target", axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



print(x)
print(y)

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled = pd.DataFrame(X_train_scaled, columns=x.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=x.columns)


print("✅ Scaled Training Data (first 5 rows):")
print(X_train_scaled.head())

print("\n✅ Scaled Testing Data (first 5 rows):")
print(X_test_scaled.head())

# You can check the mean and std to confirm
print("\nFeature Means after Scaling:", X_train_scaled.mean().round(2).tolist())
print("Feature Std after Scaling:", X_train_scaled.std().round(2).tolist())

