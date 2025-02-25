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
        if df.empty:
            raise ValueError("입력 데이터가 비어있습니다")
        
        # 날짜 정렬
        df = df.sort_values('Date')
        
        # 이상치 제거 (Z-score method)
        def remove_outliers(series, threshold=3):
            z_scores = (series - series.mean()) / series.std()
            return series[abs(z_scores) < threshold]
        
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = remove_outliers(df[col])
        
        # 결측치 처리 (deprecated 메서드 수정)
        df = df.ffill().bfill()
        
        # 기술적 지표 추가
        df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['Volume_MA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
        
        # Price_Change 계산 및 첫 행의 NaN 처리
        df['Price_Change'] = df['Close'].pct_change()
        df.loc[df.index[0], 'Price_Change'] = 0
        
        # 스케일링
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not df[features].isnull().any().any():
            self.scaler.fit(df[features])
            df[features] = self.scaler.transform(df[features])
        
        return df
    
    def prepare_train_data(self, df, sequence_length=10):
        features = ['Close', 'Volume', 'MA5', 'MA20', 'Price_Change']
        data = df[features].values
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, 0])
            
        return np.array(X), np.array(y)
