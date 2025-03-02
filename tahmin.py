import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow log seviyesini hata mesajlarına ayarla

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np
import TSLA_DUYGU  # TSLA_DUYGU.py dosyasını import et

# Veriyi yükle
data = pd.read_csv('Desktop\\train\\TSLA_2010-06-29_2025-02-13.csv')

# Duygu puanını yükle
total_sentiment_score = TSLA_DUYGU.get_total_sentiment_score()

# Hacim ve Kapanış Fiyatını Çiz
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(data['Date'], data['Volume'])
plt.title('Zaman İçinde Hacim')
plt.xlabel('Tarih')
plt.ylabel('Hacim')

plt.subplot(2, 1, 2)
plt.plot(data['Date'], data['Close'])
plt.title('Zaman İçinde Kapanış Fiyatı')
plt.xlabel('Tarih')
plt.ylabel('Kapanış Fiyatı')

plt.tight_layout()
plt.show()

# Veriyi hazırla
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
close_data = data['Close'].values.reshape(-1, 1)

# Veriyi normalize et
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Eğitim ve test veri setlerini oluştur
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Duygu puanını eğitim ve test setlerine ekle
X_train = np.hstack((X_train, np.full((X_train.shape[0], 1), total_sentiment_score)))
X_test = np.hstack((X_test, np.full((X_test.shape[0], 1), total_sentiment_score)))

# XGBoost Modeli
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
xgb_model.fit(X_train, y_train)

# Random Forest Modeli
rf_model = RandomForestRegressor(n_estimators=1000)
rf_model.fit(X_train, y_train)

# LSTM Modeli
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step + 1, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train, batch_size=1, epochs=20)

# Model tahminleri
xgb_train_predict = xgb_model.predict(X_train)
xgb_test_predict = xgb_model.predict(X_test)

rf_train_predict = rf_model.predict(X_train)
rf_test_predict = rf_model.predict(X_test)

lstm_train_predict = lstm_model.predict(X_train_lstm)
lstm_test_predict = lstm_model.predict(X_test_lstm)

# Orijinal forma geri dönüştür
xgb_train_predict = scaler.inverse_transform(xgb_train_predict.reshape(-1, 1))
xgb_test_predict = scaler.inverse_transform(xgb_test_predict.reshape(-1, 1))

rf_train_predict = scaler.inverse_transform(rf_train_predict.reshape(-1, 1))
rf_test_predict = scaler.inverse_transform(rf_test_predict.reshape(-1, 1))

lstm_train_predict = scaler.inverse_transform(lstm_train_predict)
lstm_test_predict = scaler.inverse_transform(lstm_test_predict)

# Performansı değerlendirme
xgb_train_score = np.sqrt(np.mean((xgb_train_predict - scaler.inverse_transform(y_train.reshape(-1, 1)))**2))
xgb_test_score = np.sqrt(np.mean((xgb_test_predict - scaler.inverse_transform(y_test.reshape(-1, 1)))**2))
print(f'XGBoost Train Score: {xgb_train_score:.2f} RMSE')
print(f'XGBoost Test Score: {xgb_test_score:.2f} RMSE')

rf_train_score = np.sqrt(np.mean((rf_train_predict - scaler.inverse_transform(y_train.reshape(-1, 1)))**2))
rf_test_score = np.sqrt(np.mean((rf_test_predict - scaler.inverse_transform(y_test.reshape(-1, 1)))**2))
print(f'Random Forest Train Score: {rf_train_score:.2f} RMSE')
print(f'Random Forest Test Score: {rf_test_score:.2f} RMSE')

lstm_train_score = np.sqrt(np.mean((lstm_train_predict - scaler.inverse_transform(y_train.reshape(-1, 1)))**2))
lstm_test_score = np.sqrt(np.mean((lstm_test_predict - scaler.inverse_transform(y_test.reshape(-1, 1)))**2))
print(f'LSTM Train Score: {lstm_train_score:.2f} RMSE')
print(f'LSTM Test Score: {lstm_test_score:.2f} RMSE')

# Gelecekteki tarihleri oluştur
future_dates = pd.date_range(start=data.index[-1], periods=365*3, freq='D')

# Gelecekteki fiyatları tahmin et (XGBoost)
future_predictions_xgb = []
last_60_days = scaled_data[-60:]

for _ in range(len(future_dates)):
    X_future = np.append(last_60_days.reshape(1, -1), total_sentiment_score).reshape(1, -1)
    future_price = xgb_model.predict(X_future)
    future_predictions_xgb.append(future_price[0])
    last_60_days = np.append(last_60_days[1:], future_price.reshape(1, -1), axis=0)

# Gelecekteki fiyatları tahmin et (Random Forest)
future_predictions_rf = []
last_60_days = scaled_data[-60:]

for _ in range(len(future_dates)):
    X_future = np.append(last_60_days.reshape(1, -1), total_sentiment_score).reshape(1, -1)
    future_price = rf_model.predict(X_future)
    future_predictions_rf.append(future_price[0])
    last_60_days = np.append(last_60_days[1:], future_price.reshape(1, -1), axis=0)

# Gelecekteki fiyatları tahmin et (LSTM)
future_predictions_lstm = []
last_60_days = scaled_data[-60:]

for _ in range(len(future_dates)):
    X_future = np.append(last_60_days.reshape(1, time_step, 1), total_sentiment_score).reshape(1, time_step + 1, 1)
    future_price = lstm_model.predict(X_future)
    future_predictions_lstm.append(future_price[0, 0])
    last_60_days = np.append(last_60_days[1:], future_price.reshape(1, -1), axis=0)

# Gelecekteki fiyatları orijinal forma geri dönüştür
future_predictions_xgb = scaler.inverse_transform(np.array(future_predictions_xgb).reshape(-1, 1))
future_predictions_rf = scaler.inverse_transform(np.array(future_predictions_rf).reshape(-1, 1))
future_predictions_lstm = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1))

# Ensemble tahminleri (ortalama)
future_predictions_ensemble = (future_predictions_xgb + future_predictions_rf + future_predictions_lstm) / 3

# Tahmini çiz
plt.figure(figsize=(14, 7))
plt.plot(data.index, close_data, label='Gerçek Kapanış Fiyatı')
plt.plot(data.index[time_step:len(xgb_train_predict) + time_step], xgb_train_predict, label='XGBoost Eğitim Tahmini')
plt.plot(data.index[len(xgb_train_predict) + (time_step * 2) + 1:len(close_data) - 1], xgb_test_predict, label='XGBoost Test Tahmini')
plt.plot(future_dates, future_predictions_xgb, label='XGBoost Gelecek Tahmini')
plt.plot(data.index[time_step:len(rf_train_predict) + time_step], rf_train_predict, label='Random Forest Eğitim Tahmini')
plt.plot(data.index[len(rf_train_predict) + (time_step * 2) + 1:len(close_data) - 1], rf_test_predict, label='Random Forest Test Tahmini')
plt.plot(future_dates, future_predictions_rf, label='Random Forest Gelecek Tahmini')
plt.plot(data.index[time_step:len(lstm_train_predict) + time_step], lstm_train_predict, label='LSTM Eğitim Tahmini')
plt.plot(data.index[len(lstm_train_predict) + (time_step * 2) + 1:len(close_data) - 1], lstm_test_predict, label='LSTM Test Tahmini')
plt.plot(future_dates, future_predictions_lstm, label='LSTM Gelecek Tahmini')
plt.plot(future_dates, future_predictions_ensemble, label='Ensemble Gelecek Tahmini', linestyle='--')
plt.title('Kapanış Fiyatı Tahmini')
plt.xlabel('Tarih')
plt.ylabel('Kapanış Fiyatı')
plt.legend()
plt.show()
