import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import numpy as np


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['FutureClose'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    X = df[['Return', 'MA5', 'MA10', 'MA20']]
    y = df['FutureClose']
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series):
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    rmse = np.mean(np.sqrt(-scores))
    print(f"Cross-validated RMSE: {rmse:.4f}")
    model.fit(X, y)
    return model
