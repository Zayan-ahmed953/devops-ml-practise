import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('data/data.csv')

x = df['headline']
y = df['category']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(
    max_features=200,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=100, class_weight='balanced')
model.fit(X_train_vect, Y_train)

predict = model.predict(X_test_vect)

accuracy = accuracy_score(Y_test, predict)

print(accuracy)

sample_test = [
    'Moiz is a man',
    'OF is an entertainment company',
    'Politicians should fight against corruption'
]

sample_test_vect = vectorizer.transform(sample_test)

predicted_Sample = model.predict(sample_test_vect)

print(predicted_Sample)