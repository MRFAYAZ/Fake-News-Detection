
# 📰 Fake News Detection System using Machine Learning

## 📌 Project Objective
This project aims to classify news headlines as **REAL** or **FAKE** using Natural Language Processing (NLP) and Machine Learning (ML). It also supports **live predictions** from Google News RSS feeds.

---

## ✅ Features

- Train a classifier using real (`True.csv`) and fake (`Fake.csv`) news data
- Clean and preprocess news titles using NLP (stopword removal, lowercasing)
- Use TF-IDF to vectorize text data
- Train with Logistic Regression model for classification
- Evaluate performance using accuracy and classification report
- Test real-time headlines using Google News RSS
- Save prediction results to `rss_predictions.csv`

---

## 🛠️ Tech Stack

| Technology         | Purpose                        |
|--------------------|--------------------------------|
| Python             | Programming language           |
| Pandas             | Data manipulation              |
| NLTK               | Stopword removal (NLP)         |
| Scikit-learn       | Model training and evaluation  |
| TfidfVectorizer    | Text vectorization             |
| Logistic Regression| Classification model           |
| Feedparser + BS4   | Real-time news scraping        |

---

## 🧪 Dataset

| File      | Description              |
|-----------|--------------------------|
| `Fake.csv`| Fake news articles       |
| `True.csv`| Real news articles       |

📦 Download: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## ▶️ How to Run

### 1. Install Dependencies

pip install -r requirements.txt  (bash)

Place files in the project folder:

project/
├── Fake.csv
├── True.csv
├── fake_news_rss_classifier_title_only.py
├── requirements.txt

run the script

python fake_news_rss_classifier_title_only.py

📄 Sample Output
Headline: Akhilesh Yadav backs Mamata’s NRC charge
Prediction: ✅ REAL (Confidence REAL: 0.91)

Headline: Trump says aliens will land in Mumbai mall
Prediction: ❌ FAKE (Confidence REAL: 0.15)

🧾 Saved to: rss_predictions.csv

🏁 Conclusion
This project demonstrates how machine learning models can detect fake news headlines with good accuracy. It also integrates live, real-time predictions using Google News RSS.

👨‍💻 Author
Shaik Fayaz Ahamed
B.Tech CSE – Data Science & Artificial Intelligence
Internship AI Project (June 2025)
