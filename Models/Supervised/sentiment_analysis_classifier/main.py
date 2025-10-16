import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load data
df = pd.read_csv('data/data.csv')

# Step 2: Split into features and target
X = df['ReviewText']
y = df['Sentiment']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize text
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Step 5: Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vector, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test_vector)

# Step 7: Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Try predicting new text
sample_reviews = [
    "I absolutely loved the product!",
    "It was a terrible experience.",
    "Not bad, loved it."
]

sample_vec = vectorizer.transform(sample_reviews)
predictions = model.predict(sample_vec)

print("\n🔮 Sample Predictions:")
for text, sentiment in zip(sample_reviews, predictions):
    print(f"{text} → {sentiment}")
