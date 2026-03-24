import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import spacy
import re

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

def summarize_text(text, top_n=2):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= top_n:
        return text

    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(sentences)
    scores = cosine_similarity(matrix).sum(axis=1)

    top = scores.argsort()[-top_n:]
    top.sort()
    return " ".join([sentences[i] for i in top])

def ai_summary(text):
    try:
        from transformers import pipeline
        model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        result = model(text[:1000], max_length=60, min_length=20)
        return result[0]['summary_text']
    except:
        pass

    # Fallback: smart extractive AI summary using TF-IDF + sentence scoring
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return "No content to summarize."
        if len(sentences) == 1:
            return sentences[0]

        # Score sentences by TF-IDF + position weight
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform(sentences)
        scores = cosine_similarity(matrix).sum(axis=1)

        # Boost first and last sentences (they carry key financial info)
        position_weight = [1.0] * len(sentences)
        position_weight[0] = 1.3
        position_weight[-1] = 1.1
        weighted_scores = [scores[i] * position_weight[i] for i in range(len(scores))]

        # Pick top 2-3 sentences based on length
        top_n = 3 if len(sentences) >= 5 else 2
        top_indices = sorted(
            sorted(range(len(weighted_scores)), key=lambda i: weighted_scores[i], reverse=True)[:top_n]
        )
        summary = " ".join([sentences[i] for i in top_indices])

        # Add financial context prefix
        sentiment = TextBlob(text).sentiment.polarity
        if sentiment > 0.2:
            prefix = "📈 Positive Outlook: "
        elif sentiment < -0.2:
            prefix = "📉 Cautionary Note: "
        else:
            prefix = "📊 Market Update: "

        return prefix + summary
    except Exception as e:
        return "Unable to generate summary. Please try again."

def extract_keywords(text, top_n=5):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = set(["this","that","with","from","have","were","will","their"])
    filtered = [w for w in words if w not in stop_words]

    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1

    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def detect_trend(score):
    if score > 0.2:
        return "📈 Growth"
    elif score < -0.2:
        return "📉 Decline"
    return "⚖ Stable"

def extract_entities(text):
    doc = nlp(text)
    result = []
    for ent in doc.ents:
        label = ent.label_
        if label == "ORG":
            label = "Company"
        elif label == "GPE":
            label = "Country"
        elif label == "MONEY":
            label = "Currency"
        result.append((ent.text, label))
    return result

def predict_stock_trend(sentiment, keywords):
    if sentiment > 0.3:
        return "📈 Stock Likely to Rise"
    elif sentiment < -0.3:
        return "📉 Stock Likely to Fall"
    return "⚖ Market Uncertain"

def detect_fake_news(text):
    fake_words = ["shocking","guaranteed","secret","100%","viral"]
    count = sum(1 for w in fake_words if w in text.lower())
    return "🚨 Likely Fake News" if count >= 2 else "✅ Likely Real News"