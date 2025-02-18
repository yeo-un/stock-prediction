import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.data_path = Path(__file__).parent / "dataset"
    
    def load_stock_data(self, filename="TESLA.csv"):
        file_path = self.data_path / filename
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date')
    
    def process_data(self, df):
        # 기술적 지표 추가
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Price_Change'] = df['Close'].pct_change()
        
        # 스케일링
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[features] = self.scaler.fit_transform(df[features])
        
        return df.dropna()
    
    def prepare_train_data(self, df, sequence_length=10):
        features = ['Close', 'Volume', 'MA5', 'MA20', 'Price_Change']
        data = df[features].values
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, 0])
            
        return np.array(X), np.array(y)
