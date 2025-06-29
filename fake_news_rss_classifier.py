# fake_news_rss_classifier_title_only.py

import pandas as pd
import nltk
import feedparser
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Setup
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    return " ".join([word.lower() for word in str(text).split() if word.lower() not in stop_words])

# Step 2: Load & Label
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["label"] = 0
true["label"] = 1

# Step 3: Combine & Clean (TITLE ONLY)
df = pd.concat([fake, true], ignore_index=True).sample(frac=1, random_state=42)
df["clean_text"] = df["title"].apply(clean_text)

# Step 4: Train/Test Split
X = df["clean_text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Google News RSS Prediction (Headline-Only)
print("\n📰 === LIVE GOOGLE NEWS HEADLINES ===")
rss_url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
feed = feedparser.parse(rss_url)

results = []

for i, entry in enumerate(feed.entries[:10]):
    headline = BeautifulSoup(entry.title, "html.parser").text.strip()
    print(f"\nHeadline {i+1}: {headline}")

    cleaned = clean_text(headline)
    vector = vectorizer.transform([cleaned])
    prob_real = model.predict_proba(vector)[0][1]  # Confidence REAL

    # Adjusted thresholds
    if prob_real > 0.55:
        result = "✅ REAL"
    elif prob_real < 0.45:
        result = "❌ FAKE"
    else:
        result = "🤔 UNCERTAIN"

    print(f"Prediction: {result} (Confidence REAL: {prob_real:.2f})")

    results.append({"headline": headline, "prediction": result, "confidence": round(prob_real, 2)})

# Save to CSV
pd.DataFrame(results).to_csv("rss_predictions.csv", index=False)
print("\n📁 Predictions saved to rss_predictions.csv")
