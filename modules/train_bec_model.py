import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load CEAS_08 dataset
data = pd.read_csv("datasets/CEAS_08.csv")

# Combine subject + body for richer text features
# CEAS: label 1 = spam, label 0 = ham
data["subject"] = data["subject"].fillna("")
data["body"]    = data["body"].fillna("")
data["message"] = data["subject"] + " " + data["body"]

# Map numeric labels to string — keeps bec_detector.py compatible
# 1 (spam) → "spam" | 0 (ham) → "ham"
data["label"] = data["label"].map({1: "spam", 0: "ham"})

# Drop any rows where label or message is missing
data = data.dropna(subset=["label", "message"])

print("Dataset shape:", data.shape)
print("\nLabel counts:")
print(data["label"].value_counts())

# Text and labels
X = data["message"]
y = data["label"]

# Convert text → numeric using TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,   # top 10k words — faster + avoids overfitting
    ngram_range=(1, 2)    # unigrams + bigrams — catches "wire transfer", "gift card" etc.
)
X_vec = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
acc_percent = round(acc * 100, 2)

print("\nModel Accuracy:", acc_percent, "%")

# Save accuracy
with open("models/email_accuracy", "w") as f:
    f.write(str(acc_percent))

# Save model and vectorizer
pickle.dump(model,      open("models/bec_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("BEC Email model trained successfully")