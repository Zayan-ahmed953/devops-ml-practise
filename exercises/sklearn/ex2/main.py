from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd


# TODO:
# 1. Split df into X (features) and y (target)
# 2. Use train_test_split() with test_size=0.2
# 3. Print shapes of resulting datasets


# Load the iris dataset
iris = datasets.load_iris()


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target


# 1. Split df into X (features) and y (target)
X = df.drop('target', axis=1)
y = df['target']

# print('\n\n', df)
print('\n\n', X)
print('\n\n', y)


# 2. Use train_test_split() with test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Print shapes of resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)




