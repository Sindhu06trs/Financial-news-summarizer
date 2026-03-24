from flask import Flask, render_template, request, send_file
import sqlite3
from summarizer import *

app = Flask(__name__)

conn = sqlite3.connect("database.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS history (
id INTEGER PRIMARY KEY AUTOINCREMENT,
text TEXT, summary TEXT, sentiment TEXT
)
""")

@app.route('/', methods=['GET','POST'])
def home():
    summary = ai_sum = ""
    keywords = []
    sentiment_score = 0
    sentiment_label = trend = prediction = fake_news = ""
    entities = []
    input_text = ""

    if request.method == 'POST':
        input_text = request.form.get('text')
        length = int(request.form.get('length'))

        if input_text.strip():
            summary = summarize_text(input_text, length)
            ai_sum = ai_summary(input_text)
            keywords = extract_keywords(input_text)
            sentiment_score = analyze_sentiment(input_text)
            entities = extract_entities(input_text)

            sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
            trend = detect_trend(sentiment_score)
            prediction = predict_stock_trend(sentiment_score, keywords)
            fake_news = detect_fake_news(input_text)

            cur.execute("INSERT INTO history(text,summary,sentiment) VALUES (?,?,?)",
                        (input_text, summary, sentiment_label))
            conn.commit()

    cur.execute("SELECT * FROM history ORDER BY id DESC LIMIT 5")
    history = cur.fetchall()

    return render_template("index.html",
        summary=summary,
        ai_sum=ai_sum,
        keywords=keywords,
        sentiment=sentiment_label,
        score=sentiment_score,   # ✅ FIXED
        trend=trend,
        prediction=prediction,
        fake_news=fake_news,
        entities=entities,
        history=history,
        input_text=input_text
    )

@app.route('/download', methods=['POST'])
def download():
    content = request.form['summary']
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(content)
    return send_file("report.txt", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)