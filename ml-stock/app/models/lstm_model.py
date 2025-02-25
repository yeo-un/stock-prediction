import tensorflow as tf
from keras import Sequential, Input, Model
from keras.layers import LSTM, Dense, Dropout
import numpy as np

class LSTMPredictor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = self._build_model()
        
    def _build_model(self):
        inputs = Input(shape=(self.sequence_length, 5))
        
        # LSTM 레이어 추가 및 유닛 수 증가
        x = LSTM(100, return_sequences=True)(inputs)
        x = Dropout(0.3)(x)
        x = LSTM(100, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(50)(x)
        x = Dropout(0.3)(x)
        
        # Dense 레이어 추가
        x = Dense(50, activation='relu')(x)
        x = Dense(25, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='huber',  # Huber loss는 이상치에 더 강건합니다
            metrics=['mae', 'mse']
        )
        return model
    
    def train(self, X, y, epochs=50):
        return self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
    def predict(self, X):
        return self.model.predict(X, verbose=0)
