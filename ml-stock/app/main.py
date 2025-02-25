import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.models.linear_regression import StockPredictor
from app.models.lstm_model import LSTMPredictor
from app.data.preprocessor import DataPreprocessor
import numpy as np
import plotly.graph_objects as go


class PredictionGraphResponse(BaseModel):
    graph_data: dict
    predicted_price: float

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

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 제공을 위한 디렉토리 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictionResponse(BaseModel):
    predicted_price: float

@app.get("/predict/linear")
async def predict_linear():
    try:
        # 최신 데이터 가져오기
        df = preprocessor.load_stock_data()
        processed_data = preprocessor.process_data(df)
        
        # 마지막 30일 데이터로 예측
        prediction_data = processed_data.tail(30)
        if prediction_data.empty:
            raise ValueError("예측할 데이터가 없습니다")
            
        predictions = linear_predictor.predict(prediction_data)
        
        # 원래 스케일로 되돌리기 (선택사항)
        last_prediction = float(predictions[-1])
        
        return {
            "predicted_price": last_prediction,
            "last_date": df['Date'].iloc[-1].strftime("%Y-%m-%d"),
            "current_price": float(prediction_data['Close'].iloc[-1])
        }
        
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


@app.get("/predict/graph")
async def predict_with_graph():
    try:
        # 데이터 준비
        df = preprocessor.load_stock_data()
        processed_data = preprocessor.process_data(df)
        
        # 선형 회귀와 LSTM 예측
        linear_pred = linear_predictor.predict(processed_data.tail(1))
        X, _ = preprocessor.prepare_train_data(processed_data)
        lstm_pred = lstm_predictor.predict(X[-1:])
        
        # 예측값 스케일 조정
        linear_prediction = float(linear_pred[0]) * 1000  # 선형 회귀 예측값 * 1000
        lstm_prediction = float(lstm_pred[0][0]) * 10000  # LSTM 예측값 * 10000
        
        # 날짜 데이터 준비
        last_date = df['Date'].iloc[-1]
        pred_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 최근 30일 데이터 준비
        recent_dates = df['Date'].tail(30).dt.strftime('%Y-%m-%d').tolist()
        recent_prices = df['Close'].tail(30).astype(float).tolist()
        
        fig = go.Figure()
        
        # 실제 가격 추가
        fig.add_trace(go.Scatter(
            x=recent_dates,
            y=recent_prices,
            name='실제 가격',
            mode='lines'
        ))
        
        # 예측 가격 추가
        fig.add_trace(go.Scatter(
            x=[pred_date],
            y=[linear_prediction],
            name='선형 회귀 예측',
            mode='markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=[pred_date],
            y=[lstm_prediction],
            name='LSTM 예측',
            mode='markers'
        ))
        
        # 레이아웃 설정
        fig.update_layout(
            title='주가 예측 그래프',
            xaxis_title='날짜',
            yaxis_title='가격',
            hovermode='x'
        )
        
        response_data = {
            "graph_data": fig.to_dict(),
            "predicted_price": linear_prediction  # 스케일 조정된 선형 회귀 예측값
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Error in predict_with_graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>주가 예측 그래프</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                #chart {
                    margin-top: 20px;
                }
                #prediction {
                    margin-top: 20px;
                    font-size: 18px;
                    text-align: center;
                    color: #2c3e50;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>주가 예측 그래프</h1>
                <div id="chart"></div>
                <div id="prediction"></div>
            </div>
            
            <script>
                async function fetchAndDisplayChart() {
                    try {
                        const response = await fetch('/predict/graph');
                        const data = await response.json();
                        
                        // 그래프 데이터가 있는 경우에만 그래프 그리기
                        if (data && data.graph_data) {
                            Plotly.newPlot('chart', data.graph_data.data, data.graph_data.layout);
                        }
                        
                        // 예측 가격이 있는 경우에만 표시
                        if (data && data.predicted_price !== undefined) {
                            document.getElementById('prediction').textContent = 
                                `예측 가격: ${data.predicted_price.toFixed(2)}`;
                        }
                    } catch (error) {
                        console.error('Error:', error);
                        document.getElementById('prediction').textContent = 
                            '데이터를 불러오는 중 오류가 발생했습니다.';
                    }
                }
                
                // 페이지 로드 시 그래프 표시
                fetchAndDisplayChart();
                
                // 30초마다 데이터 갱신
                setInterval(fetchAndDisplayChart, 30000);
            </script>
        </body>
    </html>
    """