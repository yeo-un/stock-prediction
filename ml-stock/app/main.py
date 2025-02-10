from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression

app = FastAPI()

class StockData(BaseModel):
    dates: list
    prices: list


@app.post('/predict')
async def predict_stock(data: StockData):
    # ML Logic
    return {'predictions':"predictions"}

