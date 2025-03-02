import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data[['Close']]
    data = data.rename(columns={'Close': 'y'})
    data['ds'] = data.index
    return data[['ds', 'y']]

if __name__ == "__main__":
    file_path = ==='
    raw_data = load_data(file_path)
    processed_data = preprocess_data(raw_data)
    print(processed_data.head())
