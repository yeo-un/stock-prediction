import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np

class LSTMPredictor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
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
