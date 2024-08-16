from data_fetcher import get_japan_economic_data
from feature_engineering import add_features, make_stationary
from models import train_lstm, train_arima, train_prophet, train_xgboost, train_random_forest
from ensemble import weighted_average_ensemble, calculate_model_weights
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def main():
    # Fetch and preprocess data
    data = get_japan_economic_data()
    data = add_features(data)
    data = make_stationary(data)
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Prepare data for modeling
    n_steps = 30
    X, y = [], []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:(i + n_steps), :])
        y.append(scaled_data[i + n_steps, 0])  # Assuming deflation is the first column
    X, y = np.array(X), np.array(y)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train models
        lstm_model = train_lstm(X_train, n_steps)
        arima_model = train_arima(data['deflation'].iloc[train_index])
        prophet_model = train_prophet(data['deflation'].iloc[train_index])
        xgb_model = train_xgboost(X_train, n_steps)
        rf_model = train_random_forest(X_train, n_steps)
        
        # Make predictions
        predictions = {
            'lstm': lstm_model.predict(X_test),
            'arima': arima_model.forecast(len(X_test)),
            'prophet': prophet_model.predict(data.index[test_index])['yhat'].values,
            'xgb': xgb_model.predict(X_test.reshape(X_test.shape[0], -1)),
            'rf': rf_model.predict(X_test.reshape(X_test.shape[0], -1))
        }
        
        # Calculate weights
        weights = calculate_model_weights(predictions, X_test, y_test)
        
        # Make ensemble prediction
        ensemble_pred = weighted_average_ensemble(predictions, weights)
        
        # Evaluate
        mae = np.mean(np.abs(ensemble_pred - y_test))
        print(f"Mean Absolute Error: {mae}")
    
    print("Forecasting completed successfully.")

if __name__ == "__main__":
    main()
