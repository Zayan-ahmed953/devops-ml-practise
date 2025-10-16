import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("data/data.csv")

X = data["subject"]
Y = data["intent"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(
    max_features=200,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = LogisticRegression( max_iter=50, class_weight='balanced' )
model.fit(X_train_vectorized, Y_train)

Predict = model.predict(X_test_vectorized)

accuracy = accuracy_score(Y_test, Predict)
print(accuracy)


sample_data = [
    "Come First Get First",
    "How much the x program cost",
    "See you at the reunion next month!",
    "You have won the lottery"
]

vector = vectorizer.transform(sample_data)

pred = model.predict(vector)
print(pred)