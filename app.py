from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

app = Flask(__name__)

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")
    return hist

def prepare_data(stock_data):
    # Basit bir özellik mühendisliği
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
    
    # NaN değerleri temizle
    stock_data = stock_data.dropna()
    
    # Hedef değişkeni oluştur (bir sonraki günün kapanış fiyatı)
    stock_data['Target'] = stock_data['Close'].shift(-1)
    
    # Son satırı kaldır çünkü hedef değeri yok
    stock_data = stock_data[:-1]
    
    return stock_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form['symbol']
    
    try:
        # Hisse senedi verilerini al
        stock_data = get_stock_data(symbol)
        
        # Verileri hazırla
        processed_data = prepare_data(stock_data)
        
        # Özellikleri ve hedef değişkeni ayır
        features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'MA5', 'MA20']
        X = processed_data[features]
        y = processed_data['Target']
        
        # Model eğitimi
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Son gün için tahmin
        last_data = X.iloc[-1:]
        prediction = model.predict(last_data)[0]
        
        # Grafik için tarih ve fiyat verilerini hazırla
        dates = stock_data.index.strftime('%Y-%m-%d').tolist()
        prices = stock_data['Close'].tolist()
        
        return jsonify({
            'success': True,
            'current_price': float(stock_data['Close'].iloc[-1]),
            'prediction': float(prediction),
            'dates': dates,
            'prices': prices
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 