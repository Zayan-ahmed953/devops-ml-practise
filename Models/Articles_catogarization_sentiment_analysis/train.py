import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('data/data.csv')

x = df['text']
y = df['category']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer (
    max_features=1000,
    stop_words='english',
    ngram_range=(1,2)
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=300, class_weight='balanced')
model.fit(X_train_vectorized, Y_train)

predict = model.predict(X_test_vectorized)

accuracy= accuracy_score(Y_test, predict)

print(accuracy)

# print("\n📊 Classification Report:\n", classification_report(Y_test, predict))

sample_test = [
    "Ronaldo is a very great sportsman",
    "Imran Khan is a Great Leader",
    "I love playing PUBG"
]

sample_test_vectorized = vectorizer.transform(sample_test)

sample_predicted = model.predict(sample_test_vectorized)

for text, sentiment in zip(sample_test, sample_predicted):
    print(f"{text} → {sentiment}")
