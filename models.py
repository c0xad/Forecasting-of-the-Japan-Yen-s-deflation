import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def prepare_data_for_lstm(data, n_steps):
    """Prepare data for LSTM model."""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), :])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build and compile LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(data, n_steps, epochs=100, batch_size=32):
    """Train LSTM model."""
    X, y = prepare_data_for_lstm(data, n_steps)
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    return model

def train_arima(data):
    """Train ARIMA model."""
    model = ARIMA(data, order=(1,1,1))  # Example order, adjust based on ACF/PACF
    return model.fit()

def train_prophet(data):
    """Train Prophet model."""
    df = pd.DataFrame({'ds': data.index, 'y': data.values})
    model = Prophet()
    model.fit(df)
    return model

def train_xgboost(data, n_steps):
    """Train XGBoost model."""
    X, y = prepare_data_for_lstm(data, n_steps)
    X = X.reshape((X.shape[0], -1))
    model = XGBRegressor()
    model.fit(X, y)
    return model

def train_random_forest(data, n_steps):
    """Train Random Forest model."""
    X, y = prepare_data_for_lstm(data, n_steps)
    X = X.reshape((X.shape[0], -1))
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    from data_fetcher import get_japan_economic_data
    from feature_engineering import add_features, make_stationary
    
    data = get_japan_economic_data()
    data = add_features(data)
    data = make_stationary(data)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    n_steps = 30
    lstm_model = train_lstm(scaled_data, n_steps)
    arima_model = train_arima(data['deflation'])
    prophet_model = train_prophet(data['deflation'])
    xgb_model = train_xgboost(scaled_data, n_steps)
    rf_model = train_random_forest(scaled_data, n_steps)
    
    print("All models trained successfully.")
