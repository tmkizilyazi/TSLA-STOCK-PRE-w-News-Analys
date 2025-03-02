import time
import json
import sys
from yahoo_fin import news
from textblob import TextBlob
import asyncio
import subprocess

# Standart çıkışın kodlamasını utf-8 olarak ayarla
sys.stdout.reconfigure(encoding='utf-8')

# Dosya adı
FILE_NAME = "TSLA_Haberler.json"
SENTIMENT_FILE = "TSLA_Sentiment.json"

# Önceden kaydedilmiş haberleri yükleme fonksiyonu
def load_existing_news():
    try:
        with open(FILE_NAME, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Haberleri kaydetme fonksiyonu
def save_news(news_data):
    with open(FILE_NAME, "w", encoding="utf-8") as file:
        json.dump(news_data, file, ensure_ascii=False, indent=4)

# Duygu analizi fonksiyonu
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# En son haberleri çekme ve analiz etme fonksiyonu
async def fetch_latest_news():
    existing_news = load_existing_news()
    existing_titles = {news['title'] for news in existing_news}
    new_entries = []
    total_sentiment_score = 0

    latest_news = news.get_yf_rss("TSLA")
    for article in latest_news:
        title = article['title']
        url = article['link']

        if title not in existing_titles:
            sentiment_score = analyze_sentiment(title)
            total_sentiment_score += sentiment_score

            # Yeni haber ekle
            new_news = {
                "title": title,
                "url": url,
                "sentiment": sentiment_score
            }
            new_entries.append(new_news)

    if new_entries:
        existing_news.extend(new_entries)
        save_news(existing_news)
        print(f"{len(new_entries)} yeni haber kaydedildi.")
        print(f"Toplam Duygu Puanı: {total_sentiment_score:.2f}")
        
        # Toplam duygu puanını kaydet
        with open(SENTIMENT_FILE, "w", encoding="utf-8") as file:
            json.dump({"total_sentiment_score": total_sentiment_score}, file, ensure_ascii=False, indent=4)
        
        # Tahmin modelini çalıştır
        subprocess.run(["python", "tahmin.py"])
    else:
        print("Yeni haber bulunamadı.")

# Duygu puanını döndüren fonksiyon
def get_total_sentiment_score():
    try:
        with open(SENTIMENT_FILE, "r", encoding="utf-8") as file:
            sentiment_data = json.load(file)
            return sentiment_data["total_sentiment_score"]
    except (FileNotFoundError, json.JSONDecodeError):
        return 0

# Haberleri çekmeye başla
if __name__ == "__main__":
    asyncio.run(fetch_latest_news())