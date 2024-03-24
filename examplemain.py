import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Attention
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_scaled_error
from sklearn.ensemble import StackingRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1_l2
from keras.layers import LayerNormalization
from tensorflow.keras.optimizers import Adam
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import shap
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
def preprocess_data(data):
"""
Preprocess the Japan Yen deflation data.
"""
data['moving_avg_5'] = data['deflation'].rolling(window=5).mean()
data['moving_avg_10'] = data['deflation'].rolling(window=10).mean()
data['moving_avg_20'] = data['deflation'].rolling(window=20).mean()
data['diff_1'] = data['deflation'].diff(periods=1)
data['diff_2'] = data['deflation'].diff(periods=2)
# Add more relevant features (e.g., economic indicators, interest rates)
# data['feature_1'] = ...
# data['feature_2'] = ...

data.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.drop(columns=['ds']))

return data_scaled, scaler

def build_stacked_lstm_model(input_shape, units, dropout, l1, l2):
"""
Build a stacked LSTM model with regularization and layer normalization.
"""
model = Sequential()
model.add(Bidirectional(LSTM(units=units, return_sequences=True, kernel_regularizer=l1_l2(l1=l1, l2=l2)), input_shape=input_shape))
model.add(LayerNormalization())
model.add(Dropout(dropout))
model.add(Bidirectional(LSTM(units=units, return_sequences=True, kernel_regularizer=l1_l2(l1=l1, l2=l2))))
model.add(LayerNormalization())
model.add(Dropout(dropout))
model.add(Bidirectional(LSTM(units=units, kernel_regularizer=l1_l2(l1=l1, l2=l2))))
model.add(LayerNormalization())
model.add(Dropout(dropout))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
return model
def build_cnn_lstm_attention_model(input_shape, units, dropout, l1, l2):
"""
Build a CNN-LSTM model with attention mechanism and regularization.
"""
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(units=units, return_sequences=True, kernel_regularizer=l1_l2(l1=l1, l2=l2))))
model.add(LayerNormalization())
model.add(Dropout(dropout))
model.add(Attention())
model.add(Bidirectional(LSTM(units=units, kernel_regularizer=l1_l2(l1=l1, l2=l2))))
model.add(LayerNormalization())
model.add(Dropout(dropout))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
return model
def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
"""
Evaluate the model's performance.
"""
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mae = mean_absolute_error(scaler.inverse_transform(y_train), scaler.inverse_transform(train_pred))
test_mae = mean_absolute_error(scaler.inverse_transform(y_test), scaler.inverse_transform(test_pred))
train_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_train), scaler.inverse_transform(train_pred))
test_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_test), scaler.inverse_transform(test_pred))
train_r2 = r2_score(scaler.inverse_transform(y_train), scaler.inverse_transform(train_pred))
test_r2 = r2_score(scaler.inverse_transform(y_test), scaler.inverse_transform(test_pred))
train_mse = mean_squared_error(scaler.inverse_transform(y_train), scaler.inverse_transform(train_pred))
test_mse = mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(test_pred))
train_mase = mean_absolute_scaled_error(scaler.inverse_transform(y_train), scaler.inverse_transform(train_pred))
test_mase = mean_absolute_scaled_error(scaler.inverse_transform(y_test), scaler.inverse_transform(test_pred))

print("Training Loss:", train_loss)
print("Test Loss:", test_loss)
print("Training MAE:", train_mae)
print("Test MAE:", test_mae)
print("Training MAPE:", train_mape)
print("Test MAPE:", test_mape)
print("Training R-squared:", train_r2)
print("Test R-squared:", test_r2)
print("Training MSE:", train_mse)
print("Test MSE:", test_mse)
print("Training MASE:", train_mase)
print("Test MASE:", test_mase)

return train_loss, test_loss, train_mae, test_mae, train_mape, test_mape, train_r2, test_r2, train_mse, test_mse, train_mase, test_mase

def make_ensemble_forecast(best_model, arima_model, prophet_model, xgb_model, rf_model, data_scaled, scaler, future_steps):
"""
Make ensemble forecast using the best LSTM model, ARIMA, Prophet, XGBoost, and Random Forest.
"""
future_data = data_scaled[-time_steps:]
future_data = np.reshape(future_data, (1, time_steps, future_data.shape
1
))
future_lstm_predictions = best_model.predict(future_data)
future_lstm_predictions = scaler.inverse_transform(future_lstm_predictions.reshape(-1, 1))
future_arima_predictions = arima_model.forecast(steps=future_steps)[0]

future_prophet_predictions = prophet_model.predict(prophet_model.make_future_dataframe(periods=future_steps))['yhat'].values[-future_steps:]

future_xgb_predictions = xgb_model.predict(future_data.reshape(future_data.shape[0], -1))
future_xgb_predictions = scaler.inverse_transform(future_xgb_predictions.reshape(-1, 1))

future_rf_predictions = rf_model.predict(future_data.reshape(future_data.shape[0], -1))
future_rf_predictions = scaler.inverse_transform(future_rf_predictions.reshape(-1, 1))

ensemble_predictions = (future_lstm_predictions.reshape(-1, 1) + future_arima_predictions + future_prophet_predictions + future_xgb_predictions + future_rf_predictions) / 5

return ensemble_predictions

def explain_model_predictions(model, X_test, scaler):
"""
Use SHAP to explain the model's predictions.
"""
explainer = shap.DeepExplainer(model, X_test)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values
0
, scaler.inverse_transform(X_test.reshape(X_test.shape
0
, -1)), plot_type="bar", feature_names=data.columns[1:])
Load the data
data = pd.read_csv('japan_yen_deflation.csv')
Preprocess the data
data_scaled, scaler = preprocess_data(data)
Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled[:-10], data_scaled[10:], test_size=0.2, shuffle=False)
Reshape the data for LSTM input
time_steps = 30
X_train = np.reshape(X_train, (X_train.shape
0
, time_steps, X_train.shape
1
))
X_test = np.reshape(X_test, (X_test.shape
0
, time_steps, X_test.shape
1
))
Hyperparameter tuning using Bayesian optimization
param_space = {
'units': Integer(64, 256),
'dropout': Real(0.2, 0.5),
'l1': Real(1e-5, 1e-3, prior='log-uniform'),
'l2': Real(1e-5, 1e-3, prior='log-uniform'),
'batch_size': Integer(16, 64),
'epochs': Integer(100, 200)
}
bayes_search = BayesSearchCV(
estimator=build_stacked_lstm_model((X_train.shape
1
, X_train.shape
2
), 128, 0.3, 1e-4, 1e-4),
search_spaces=param_space,
n_iter=20,
cv=TimeSeriesSplit(n_splits=3),
scoring='neg_mean_squared_error',
verbose=1
)
bayes_search.fit(X_train, y_train)
Get the best hyperparameters
best_params = bayes_search.best_params_
print("Best Hyperparameters:", best_params)
Retrain the model with the best hyperparameters
best_model = build_stacked_lstm_model((X_train.shape
1
, X_train.shape
2
), best_params['units'], best_params['dropout'], best_params['l1'], best_params['l2'])
best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_data=(X_test, y_test),
callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)])
Evaluate the best model
(train_loss, test_loss, train_mae, test_mae, train_mape, test_mape, train_r2, test_r2,train_mse, test_mse, train_mase, test_mase) = evaluate_model(best_model, X_train, y_train, X_test, y_test, scaler)
Visualize the model's performance
plt.figure(figsize=(12, 6))
plt.plot(best_model.history.history['loss'], label='Training Loss')
plt.plot(best_model.history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(data_scaled[:, 0]), label='Actual Deflation')
plt.plot(scaler.inverse_transform(best_model.predict(X_test)), label='Predicted Deflation')
plt.title('Actual vs. Predicted Deflation')
plt.xlabel('Time')
plt.ylabel('Deflation')
plt.legend()
plt.show()
Make future predictions
future_steps = 10
arima_model = ARIMA(data['deflation'], order=(1, 1, 1))
arima_model_fit = arima_model.fit()
prophet_model = Prophet()
prophet_model.fit(data[['ds', 'deflation']])
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train.reshape(X_train.shape
0
, -1), y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train.reshape(X_train.shape
0
, -1), y_train)
ensemble_predictions = make_ensemble_forecast(best_model, arima_model_fit, prophet_model, xgb_model, rf_model, data_scaled, scaler, future_steps)
print("Ensemble Forecast for the next 10 steps:")
print(ensemble_predictions)
Evaluate the ensemble model
ensemble_mae = mean_absolute_error(scaler.inverse_transform(y_test), scaler.inverse_transform(ensemble_predictions))
ensemble_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_test), scaler.inverse_transform(ensemble_predictions))
ensemble_r2 = r2_score(scaler.inverse_transform(y_test), scaler.inverse_transform(ensemble_predictions))
ensemble_mase = mean_absolute_scaled_error(scaler.inverse_transform(y_test), scaler.inverse_transform(ensemble_predictions))
print("Ensemble MAE:", ensemble_mae)
print("Ensemble MAPE:", ensemble_mape)
print("Ensemble R-squared:", ensemble_r2)
print("Ensemble MASE:", ensemble_mase)
Explain the model's predictions using SHAP
explain_model_predictions(best_model, X_test, scaler)
