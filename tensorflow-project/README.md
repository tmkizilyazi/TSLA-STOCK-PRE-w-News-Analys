# TensorFlow Stock Price Prediction Project

This project implements a stock price prediction model for Tesla (TSLA) using TensorFlow. The model is trained on historical stock price data and can make future price predictions.

## Project Structure

```
tensorflow-project
├── data
│   └── TSLA_2010-06-29_2025-02-13.csv  # Historical stock price data for Tesla
├── src
│   ├── data_preprocessing.py            # Data loading and preprocessing
│   ├── model_training.py                 # Model definition and training
│   └── prediction.py                     # Making predictions with the trained model
├── requirements.txt                      # Project dependencies
└── README.md                             # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd tensorflow-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Preprocess the data:
   Run the `data_preprocessing.py` script to load and preprocess the data.
   ```
   python src/data_preprocessing.py
   ```

2. Train the model:
   Execute the `model_training.py` script to define and train the TensorFlow model.
   ```
   python src/model_training.py
   ```

3. Make predictions:
   Use the `prediction.py` script to load the trained model and make predictions.
   ```
   python src/prediction.py
   ```

## Overview

This project aims to provide a simple yet effective way to predict stock prices using deep learning techniques. The model leverages historical data to forecast future prices, which can be useful for investors and analysts.