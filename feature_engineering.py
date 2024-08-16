import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def add_features(df):
    """
    Add engineered features to the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with economic indicators.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional engineered features.
    """
    df['deflation_lag1'] = df['deflation'].shift(1)
    df['deflation_lag7'] = df['deflation'].shift(7)
    df['deflation_lag30'] = df['deflation'].shift(30)
    
    df['usdjpy_pct_change'] = df['usdjpy'].pct_change()
    df['oil_price_pct_change'] = df['oil_price'].pct_change()
    
    df['deflation_ma7'] = df['deflation'].rolling(window=7).mean()
    df['deflation_ma30'] = df['deflation'].rolling(window=30).mean()
    
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    
    return df.dropna()

def check_stationarity(series):
    """
    Check stationarity of a time series using Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    series : pd.Series
        Input time series.
    
    Returns:
    --------
    bool
        True if series is stationary, False otherwise.
    """
    result = adfuller(series.dropna())
    return result[1] <= 0.05  # p-value threshold

def make_stationary(df):
    """
    Transform non-stationary series to stationary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with stationary series.
    """
    for column in df.columns:
        if not check_stationarity(df[column]):
            df[f'{column}_diff'] = df[column].diff()
    return df.dropna()

if __name__ == "__main__":
    from data_fetcher import get_japan_economic_data
    
    data = get_japan_economic_data()
    data_with_features = add_features(data)
    stationary_data = make_stationary(data_with_features)
    
    print(stationary_data.head())
    print(stationary_data.shape)
