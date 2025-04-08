from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler
from news_analyzer import NewsAnalyzer
import json

app = Flask(__name__)

def create_kalman_filter():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.], [0.]])  # state
    kf.F = np.array([[1., 1.],
                     [0., 1.]])    # state transition matrix
    kf.H = np.array([[1., 0.]])    # measurement function
    kf.P *= 10.                    # covariance matrix (çok düşük başlangıç belirsizliği)
    kf.R = 0.01                    # measurement noise (çok düşük ölçüm gürültüsü)
    kf.Q = np.array([[0.001, 0.001],
                     [0.001, 0.001]]) # process noise (çok düşük süreç gürültüsü)
    return kf

def apply_kalman_filter(prices):
    kf = create_kalman_filter()
    filtered_prices = []
    
    # İlk değeri ayarla
    kf.x[0] = prices[0]
    
    # Hareketli ortalama hesapla (5 günlük)
    ma5 = pd.Series(prices).rolling(window=5).mean()
    
    for i, price in enumerate(prices):
        kf.predict()
        
        # Eğer hareketli ortalama mevcutsa, ölçümü ona göre ayarla
        if not np.isnan(ma5[i]):
            # Fiyat hareketli ortalamadan çok uzaksa, ölçümü düzelt
            if abs(price - ma5[i]) > ma5[i] * 0.02:  # %2'den fazla sapma varsa
                price = ma5[i]
        
        kf.update(price)
        filtered_prices.append(kf.x[0])
    
    return np.array(filtered_prices)

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:(i + seq_length)])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

def create_lstm_model(seq_length, n_features):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='mse',
                 metrics=['mae', 'mse'])
    return model

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    # Tüm geçmiş veriyi çek
    hist = stock.history(period="max")
    return hist

def prepare_data(hist, symbol):
    # Kalman filtresi uygula
    filtered_close = apply_kalman_filter(hist['Close'].values)
    hist['Filtered_Close'] = filtered_close

    # Teknik göstergeler
    hist['MA5'] = hist['Close'].rolling(window=5).mean()
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    
    # Haber analizi
    news_analyzer = NewsAnalyzer()
    news_features = news_analyzer.get_news_features(symbol)
    
    # Haber özelliklerini ekle (etkisini daha da azalt)
    hist['News_Sentiment'] = news_features['news_sentiment'] * 0.15  # %15 etki
    hist['Has_Recent_News'] = news_features['has_recent_news'] * 0.1  # %10 etki
    
    # NaN değerleri temizle
    hist = hist.dropna()
    
    return hist

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        symbol = request.form['symbol']
        hist = get_stock_data(symbol)
        
        if hist.empty:
            return jsonify({'success': False, 'error': 'Veri bulunamadı'})
        
        # Haber analizini yap
        news_analyzer = NewsAnalyzer()
        news_features = news_analyzer.get_news_features(symbol)
        
        # Veriyi hazırla
        processed_data = prepare_data(hist, symbol)
        
        # Random Forest için özellikler
        features_rf = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'News_Sentiment', 'Has_Recent_News']
        X_rf = processed_data[features_rf].values
        y_rf = processed_data['Close'].values
        
        # LSTM için veri hazırlama
        seq_length = 30
        n_features = 7
        features_lstm = ['Open', 'High', 'Low', 'Volume', 'Filtered_Close', 'News_Sentiment', 'Has_Recent_News']
        data_lstm = processed_data[features_lstm].values
        
        # Veriyi normalize et
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_lstm)
        
        # Sequence'leri oluştur
        X_lstm, y_lstm = create_sequences(data_scaled, seq_length)
        
        # Random Forest modeli
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=400, random_state=42)
        rf_model.fit(X_train_rf, y_train_rf)
        
        # LSTM modeli
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
        lstm_model = create_lstm_model(seq_length, n_features)
        
        # Early stopping ve learning rate scheduler ekle
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )
        
        # Model eğitimi
        history = lstm_model.fit(
            X_train_lstm, y_train_lstm,
            epochs=200,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Son veriyi al ve tahmin yap
        last_data_rf = X_rf[-1:]
        last_data_lstm = X_lstm[-1:]
        
        rf_pred = rf_model.predict(last_data_rf)[0]
        lstm_pred = lstm_model.predict(last_data_lstm)[0][0]
        
        # Tahminleri birleştir
        prediction = (rf_pred * 0.8 + lstm_pred * 0.2)
        
        # Tahmin tarihini hesapla
        last_date = processed_data.index[-1]
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:  # Hafta sonu kontrolü
            next_date += timedelta(days=1)
        
        # Sonuçları hazırla
        result = {
            'success': True,
            'current_price': float(hist['Close'].iloc[-1]),
            'prediction': float(prediction),
            'prediction_date': next_date.strftime('%Y-%m-%d'),
            'dates': processed_data.index.strftime('%Y-%m-%d').tolist(),
            'prices': processed_data['Close'].tolist(),
            'filtered_prices': processed_data['Filtered_Close'].tolist(),
            'news_sentiment': float(news_features['news_sentiment'])  # Haber etkisini ekle
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 