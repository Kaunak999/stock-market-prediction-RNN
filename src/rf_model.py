import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from utils_rnn import plot_seqs


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mda(y_true, y_pred, t=12):
    return np.mean(np.sign(y_true[t:] - y_true[:-t]) == np.sign(y_pred[t:] - y_pred[:-t]))


class RandomForestStockPredictor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        self.scaler = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        self.fit(X_train, y_train)
        y_train_pred = self.predict(X_train)
        y_val_pred = self.predict(X_val)

        train_rmse = rmse(y_train, y_train_pred)
        val_rmse = rmse(y_val, y_val_pred)
        train_mda = mda(y_train, y_train_pred)
        val_mda = mda(y_val, y_val_pred)

        print(f"Train RMSE: {train_rmse:.4f}, Train MDA: {train_mda:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}, Validation MDA: {val_mda:.4f}")

        return y_train_pred, y_val_pred


def preprocess_data(data, target_col, window_size=12, split_ratio=0.8, scaler="standard"):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    split_idx = int(len(X) * split_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    if scaler == "standard":
        sc = StandardScaler()
    elif scaler == "minmax":
        sc = MinMaxScaler()

    X_train = sc.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = sc.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    y_train = sc.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val = sc.transform(y_val.reshape(-1, 1)).reshape(-1)

    return X_train, y_train, X_val, y_val, sc


def plot_predictions(true, pred, title, labels=None):
    plt.figure(figsize=(12, 6))
    plt.plot(true, label="True", alpha=0.7)
    plt.plot(pred, label="Predicted", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend(labels if labels else ["True", "Predicted"])
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Load your stock market data
    # Example: data = pd.read_csv('stock_data.csv')
    np.random.seed(42)
    data = np.cumsum(np.random.randn(1000))  # Dummy data simulating stock prices

    # Preprocess the data
    X_train, y_train, X_val, y_val, scaler = preprocess_data(data, target_col=None, window_size=12)

    # Initialize and train the model
    rf_predictor = RandomForestStockPredictor(n_estimators=100, max_depth=10)
    y_train_pred, y_val_pred = rf_predictor.train_and_evaluate(X_train, y_train, X_val, y_val)

    # Rescale predictions back to original scale for visualization
    y_val_rescaled = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
    y_val_pred_rescaled = scaler.inverse_transform(y_val_pred.reshape(-1, 1)).reshape(-1)

    # Plot predictions
    plot_predictions(y_val_rescaled, y_val_pred_rescaled, "Random Forest Stock Prediction")
