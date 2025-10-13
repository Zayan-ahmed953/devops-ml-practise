from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score




# TODO:
# 1. Load california housing dataset
# 2. Train LinearRegression model
# 3. Evaluate using mean_squared_error

housing_dataset = fetch_california_housing()

x = housing_dataset.data
y = housing_dataset.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scale = StandardScaler()
scale.fit(X_train)

X_train_scaled = scale.transform(X_train)
X_test_scaled = scale.transform(X_test)


lnn = LinearRegression()
lnn.fit(X_train_scaled, y_train)

predict = lnn.predict(X_test_scaled)

mse = mean_squared_error(y_test, predict)


r2 = r2_score(y_test, predict)


print(r2)
print(mse)