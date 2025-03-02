import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('path_to_your_trained_model.h5')

# Load and preprocess the data
data = pd.read_csv('data/TSLA_2010-06-29_2025-02-13.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Prepare the input data for prediction
# Assuming we want to predict the next day's closing price
last_days = data['Close'].values[-30:]  # Use the last 30 days for prediction
last_days = last_days.reshape((1, 30, 1))  # Reshape for LSTM input

# Make predictions
predicted_price = model.predict(last_days)
predicted_price = predicted_price.flatten()

# Output the predicted stock price
print(f'Predicted stock price for the next day: {predicted_price[0]}')