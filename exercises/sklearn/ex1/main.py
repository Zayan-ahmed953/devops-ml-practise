from sklearn import datasets
import pandas as pd

# Load the iris dataset
iris = datasets.load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target


# TODO:
# 1. Print first 5 rows
# 2. Print shape and data types
# 3. Count how many samples of each class exist


print(df.head(5))
print(df.shape)
print(df.dtypes)

print('\n\n\n\n')
print(df['target'].value_counts())