# Task 1: Spam Email Classifier
# Developed by: Shreenidhi B S
# Internship: KODBUD AI Internship (AICTE Approved)

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Load Dataset
# You can download 'spam.csv' from Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 2️⃣ Data Preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3️⃣ Split Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 4️⃣ Text Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5️⃣ Train Model using Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6️⃣ Make Predictions
y_pred = model.predict(X_test_tfidf)

# 7️⃣ Evaluate Model
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8️⃣ Test with Custom Messages
sample_messages = [
    "Congratulations! You have won a free lottery worth $1000. Call now!",
    "Hey, are we still meeting for lunch today?",
    "You have been selected for a special offer. Click the link to claim."
]

sample_features = vectorizer.transform(sample_messages)
predictions = model.predict(sample_features)

print("\n🔍 Sample Predictions:")
for msg, pred in zip(sample_messages, predictions):
    label = "Spam" if pred == 1 else "Not Spam"
    print(f"Message: {msg}\nPrediction: {label}\n")
