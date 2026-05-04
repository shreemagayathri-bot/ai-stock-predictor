import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def prepare_data(df, seq_len=60, pred_len=7):
    """
    Normalizes data and creates sequences for LSTM.

    Returns: X, y, scaler
    """
    data = df[['Open', 'High', 'Low', 'Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    for i in range(seq_len, len(scaled_data) - pred_len + 1):
        X.append(scaled_data[i - seq_len:i])
        # Predicting 'Close' price (index 3 of features)
        y.append(scaled_data[i:i + pred_len, 3])

    return np.array(X), np.array(y), scaler


def build_lstm_model(seq_len, pred_len):
    """
    Builds the Keras LSTM model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 4)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(pred_len)  # Output matches prediction days
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


def predict_future(model, last_sequence, scaler):
    """
    Runs inference on the latest sequence.
    """
    prediction = model.predict(last_sequence.reshape(1, -1, 4))

    # Create dummy array to perform inverse_transform
    # scaler expects 4 features (OHLC), so we fill only Close
    dummy = np.zeros((prediction.shape[1], 4))
    dummy[:, 3] = prediction[0]

    inverse_pred = scaler.inverse_transform(dummy)[:, 3]
    return inverse_pred