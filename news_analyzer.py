import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
import numpy as np
from datetime import datetime, timedelta

class NewsAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_news(self, symbol):
        """Yahoo Finance'den son haberleri çeker"""
        try:
            # Yahoo Finance haber URL'si
            url = f'https://finance.yahoo.com/quote/{symbol}/news'
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Haber başlıklarını ve tarihlerini çek
            news_items = []
            for item in soup.find_all('h3', class_='Mb(5px)')[:10]:  # Son 10 haber
                title = item.text.strip()
                link = item.find('a')['href']
                if not link.startswith('http'):
                    link = 'https://finance.yahoo.com' + link
                news_items.append({
                    'title': title,
                    'link': link
                })
            
            return news_items
        except Exception as e:
            print(f"Haber çekme hatası: {str(e)}")
            return []

    def analyze_sentiment(self, text):
        """Metin duygu analizi yapar"""
        try:
            analysis = TextBlob(text)
            # -1 ile 1 arasında bir skor döndürür
            return analysis.sentiment.polarity
        except:
            return 0

    def get_news_sentiment(self, symbol):
        """Haberlerin duygu analizini yapar ve bir skor döndürür"""
        news_items = self.get_news(symbol)
        if not news_items:
            return 0
        
        sentiments = []
        for item in news_items:
            sentiment = self.analyze_sentiment(item['title'])
            sentiments.append(sentiment)
        
        # Ortalama duygu skoru
        avg_sentiment = np.mean(sentiments)
        return avg_sentiment

    def get_news_features(self, symbol):
        """Haber analizinden özellikler çıkarır"""
        sentiment_score = self.get_news_sentiment(symbol)
        
        # Duygu skorunu -1 ile 1 arasından 0 ile 1 arasına normalize et
        normalized_sentiment = (sentiment_score + 1) / 2
        
        return {
            'news_sentiment': normalized_sentiment,
            'has_recent_news': 1 if sentiment_score != 0 else 0
        } 