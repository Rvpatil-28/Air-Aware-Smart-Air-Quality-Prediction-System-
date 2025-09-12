"""
AirAware - Module 2: Model Training & Evaluation (Final Version)

Trains ARIMA, Prophet, and LSTM on a pollutant time series.
Evaluates with MAE and RMSE, selects the best model, and saves results.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle
import matplotlib.pyplot as plt

# --- Imports ---
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except Exception:
        PROPHET_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# -------------------------
# 1. PARAMETERS / PATHS
# -------------------------
DATA_CSV = "Air_Quality.py/synthetic_air_quality_1yr.csv"  # replace with your dataset
TARGET = "PM2.5"                            # pollutant to forecast
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TEST_DAYS = 14        # forecast horizon
LSTM_TIMESTEPS = 14   # sequence length
LSTM_EPOCHS = 50
LSTM_BATCH = 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------
# 2. PREPROCESSING
# -------------------------
def preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.interpolate(method='time').bfill().ffill()
    for col in ['PM2.5','PM10','NO2','SO2','O3']:
        if col in df.columns:
            q_low, q_high = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = np.clip(df[col], q_low, q_high)
    df_daily = df.resample('D').mean()
    return df_daily

print("Loading and preprocessing data...")
df_daily = preprocess(DATA_CSV)
series = df_daily[[TARGET]].dropna()

train = series.iloc[:-TEST_DAYS]
test = series.iloc[-TEST_DAYS:]

print(f"Data length: {len(series)} days, train: {len(train)}, test: {len(test)}")

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

results = {}

# -------------------------
# 3. ARIMA
# -------------------------
print("\n--- Training ARIMA ---")
try:
    if PMDARIMA_AVAILABLE:
        arima_model = auto_arima(train[TARGET].values, seasonal=False, stepwise=True,
                                 suppress_warnings=True, error_action='ignore')
        arima_forecast = arima_model.predict(n_periods=len(test))
        arima_pred = pd.Series(arima_forecast, index=test.index)
        path = os.path.join(MODEL_DIR, f"arima_auto_{TARGET}.pkl")
        arima_model.save(path)
    else:
        arima_sm = ARIMA(train[TARGET].values, order=(5,1,0)).fit()
        arima_forecast = arima_sm.forecast(steps=len(test))
        arima_pred = pd.Series(arima_forecast, index=test.index)
        path = os.path.join(MODEL_DIR, f"arima_sm_{TARGET}.pkl")
        with open(path, "wb") as f:
            pickle.dump(arima_sm, f)
    mae, rmse = evaluate(test[TARGET].values, arima_pred.values)
    results['ARIMA'] = {'mae': mae, 'rmse': rmse, 'pred': arima_pred, 'path': path}
    print(f"ARIMA MAE={mae:.3f} RMSE={rmse:.3f} saved -> {path}")
except Exception as e:
    print("ARIMA training failed:", e)

# -------------------------
# 4. Prophet
# -------------------------
if PROPHET_AVAILABLE:
    print("\n--- Training Prophet ---")
    prophet_train = train.reset_index().rename(columns={'datetime':'ds', TARGET:'y'})
    try:
        m = Prophet()
        m.fit(prophet_train)
        future = m.make_future_dataframe(periods=len(test), freq='D')
        forecast = m.predict(future)
        prophet_pred = forecast.set_index('ds')['yhat'].loc[test.index]
        mae, rmse = evaluate(test[TARGET].values, prophet_pred.values)
        path = os.path.join(MODEL_DIR, f"prophet_{TARGET}.pkl")
        with open(path, "wb") as f:
            pickle.dump(m, f)
        results['Prophet'] = {'mae': mae, 'rmse': rmse, 'pred': prophet_pred, 'path': path}
        print(f"Prophet MAE={mae:.3f} RMSE={rmse:.3f} saved -> {path}")
    except Exception as e:
        print("Prophet training failed:", e)
else:
    print("Prophet not available.")

# -------------------------
# 5. LSTM
# -------------------------
if TENSORFLOW_AVAILABLE:
    print("\n--- Training LSTM ---")
    values = series[TARGET].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{TARGET}.save")
    joblib.dump(scaler, scaler_path)

    def create_sequences(data, timesteps):
        X, y = [], []
        for i in range(len(data)-timesteps):
            X.append(data[i:i+timesteps, 0])
            y.append(data[i+timesteps, 0])
        return np.array(X), np.array(y)

    X_all, y_all = create_sequences(scaled, LSTM_TIMESTEPS)
    split_index = len(series) - TEST_DAYS - LSTM_TIMESTEPS
    if split_index <= 0: split_index = int(len(X_all)*0.8)
    X_train, y_train = X_all[:split_index], y_all[:split_index]
    X_test, y_test = X_all[split_index:], y_all[split_index:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(64, input_shape=(LSTM_TIMESTEPS, 1)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
              validation_data=(X_test, y_test), callbacks=[es], verbose=1)

    lstm_preds_scaled = model.predict(X_test).flatten()
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled.reshape(-1,1)).flatten()
    y_test_inversed = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    first_pred_idx = LSTM_TIMESTEPS + split_index
    lstm_index = series.index[first_pred_idx:first_pred_idx+len(lstm_preds)]
    lstm_pred_series = pd.Series(lstm_preds, index=lstm_index)

    mae, rmse = evaluate(y_test_inversed, lstm_preds)
    lstm_model_path = os.path.join(MODEL_DIR, f"lstm_{TARGET}.keras")  # ✅ fixed extension
    model.save(lstm_model_path)

    results['LSTM'] = {'mae': mae, 'rmse': rmse, 'pred': lstm_pred_series,
                       'path': lstm_model_path, 'scaler': scaler_path}
    print(f"LSTM MAE={mae:.3f} RMSE={rmse:.3f} saved -> {lstm_model_path}")
else:
    print("TensorFlow not available; skipping LSTM.")

# -------------------------
# 6. SELECT BEST MODEL
# -------------------------
print("\n--- Summary of results ---")
for name, info in results.items():
    print(f"{name}: MAE={info['mae']:.3f} RMSE={info['rmse']:.3f}")

best = min(results.items(), key=lambda kv: kv[1]['rmse'])
best_name, best_info = best
print(f"\nBest model: {best_name} (RMSE={best_info['rmse']:.3f})")

meta = {
    'target': TARGET,
    'test_days': TEST_DAYS,
    'best_model': best_name,
    'metrics': {k:{'mae':v['mae'],'rmse':v['rmse'],'path':v['path']} for k,v in results.items()}
}
meta_path = os.path.join(MODEL_DIR, f"model_meta_{TARGET}.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)
print(f"Metadata saved -> {meta_path}")

# -------------------------
# 7. PLOT RESULTS
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(test.index, test[TARGET].values, marker='o', label='True (test)')
for name in results:
    pred = results[name]['pred']
    plt.plot(pred.index, pred.values, linestyle='--', label=f"{name} pred")
plt.title(f"{TARGET} - True vs Predictions")
plt.legend()
plt.show()

print("\nModule 2 complete ✅")
