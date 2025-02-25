import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_columns = None  # 학습에 사용된 특성 컬럼 저장
        
    def prepare_features(self, df):
        if df.empty:
            raise ValueError("입력 데이터가 비어있습니다")
        
        try:
            features = df[['Close', 'Volume', 'Open', 'High', 'Low']].copy()
            
            # 결측치 처리
            features = features.ffill().bfill()
            
            # RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            features['RSI'] = 100 - (100 / (1 + rs))
            
            # 이동평균
            features['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
            features['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
            features['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            
            # 가격 변화율
            features['Price_Change'] = df['Close'].pct_change()
            features.loc[features.index[0], 'Price_Change'] = 0
            
            # 거래량 지표
            features['Volume_MA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
            features['Volume_Ratio'] = df['Volume'] / features['Volume_MA5'].replace(0, np.nan)
            
            # NaN 및 무한값 처리
            features = features.replace([np.inf, -np.inf], 0)
            features = features.ffill().bfill()
            
            # 최종 NaN 체크 및 처리
            if features.isnull().any().any():
                print("NaN이 있는 컬럼:", features.columns[features.isnull().any()].tolist())
                features = features.fillna(0)
            
            return features
            
        except Exception as e:
            print(f"Feature preparation error: {str(e)}")
            print("사용 가능한 컬럼:", features.columns.tolist())  # 디버깅을 위한 컬럼 목록 출력
            raise
    
    def train(self, data):
        try:
            features = self.prepare_features(data)
            # Close를 제외한 모든 컬럼을 특성으로 사용
            self.feature_columns = [col for col in features.columns if col != 'Close']
            X = features[self.feature_columns]
            y = features['Close']
            
            print("학습에 사용되는 특성:", self.feature_columns)  # 디버깅을 위한 특성 목록 출력
            
            # 스케일러 학습 및 변환
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.fit(X_scaled, y)
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
    
    def predict(self, data):
        try:
            if self.feature_columns is None:
                raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
                
            features = self.prepare_features(data)
            X = features[self.feature_columns]
            
            print("예측에 사용되는 특성:", self.feature_columns)  # 디버깅을 위한 특성 목록 출력
            
            # 스케일링
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise
