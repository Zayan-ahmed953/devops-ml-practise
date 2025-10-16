import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('data/data.csv', quotechar='"')

x = data['text']
y = data['emotion']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vertizer = TfidfVectorizer(
    max_features=200,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vertor = vertizer.fit_transform(X_train)
X_test_vertor = vertizer.transform(X_test)

model = LogisticRegression(max_iter=100, class_weight="balanced")
model.fit(X_train_vertor, Y_train)

pred = model.predict(X_test_vertor)

accuracy = accuracy_score(Y_test, pred)
print(accuracy)

sample_data = [
    "Sadly I couldnt pass my test",
    "Our team won",
    "O really, Thats for me"
]

sample_data_vector = vertizer.transform(sample_data)

predicted_sample = model.predict(sample_data_vector)
print(predicted_sample)