import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        features = df[['Close', 'Volume']].copy()
        features['MA5'] = df['Close'].rolling(window=5).mean()
        features['MA20'] = df['Close'].rolling(window=20).mean()
        features['Price_Change'] = df['Close'].pct_change()
        return features.dropna()
    
    def train(self, data):
        features = self.prepare_features(data)
        X = features[['Volume', 'MA5', 'MA20', 'Price_Change']]
        y = features['Close']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, data):
        features = self.prepare_features(data)
        X = features[['Volume', 'MA5', 'MA20', 'Price_Change']]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
