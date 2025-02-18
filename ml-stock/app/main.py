from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.models.linear_regression import StockPredictor
from app.models.lstm_model import LSTMPredictor
from app.data.preprocessor import DataPreprocessor
import numpy as np

# 전역 변수로 모델과 전처리기 선언
preprocessor = DataPreprocessor()
linear_predictor = StockPredictor()
lstm_predictor = LSTMPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    try:
        # 데이터 로드 및 전처리
        df = preprocessor.load_stock_data()
        processed_data = preprocessor.process_data(df)
        
        # 선형 회귀 모델 학습
        linear_predictor.train(processed_data)
        
        # LSTM 모델 학습
        X, y = preprocessor.prepare_train_data(processed_data)
        lstm_predictor.train(X, y)
        print("Models trained successfully")
    except Exception as e:
        print(f"Startup error: {str(e)}")
    
    yield
    
    # 종료 시 실행
    print("Shutting down application")

app = FastAPI(lifespan=lifespan)

class PredictionResponse(BaseModel):
    predicted_price: float

@app.get("/predict/linear", response_model=PredictionResponse)
async def predict_linear():
    try:
        df = preprocessor.load_stock_data()
        processed_data = preprocessor.process_data(df)
        prediction = linear_predictor.predict(processed_data.tail(1))
        return PredictionResponse(predicted_price=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/lstm", response_model=PredictionResponse)
async def predict_lstm():
    try:
        df = preprocessor.load_stock_data()
        processed_data = preprocessor.process_data(df)
        X, _ = preprocessor.prepare_train_data(processed_data)
        prediction = lstm_predictor.predict(X[-1:])
        return PredictionResponse(predicted_price=float(prediction[0][0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
