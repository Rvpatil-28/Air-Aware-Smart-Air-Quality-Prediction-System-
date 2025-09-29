
import os
import pickle
import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow imports for LSTM loading
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# For scaler (saved by joblib in Module 2)
try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

# -------------------------
# CONFIG
# -------------------------
DATA_CSV = "Air_Quality.py/synthetic_air_quality_1yr.csv"   # replace with your daily CSV
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
TARGET = "PM2.5"         # primary pollutant to visualize & compute AQI
FORECAST_HORIZON = 14    # days to forecast / evaluate (same as Module 2)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# AQI alert threshold (category Unhealthy and above). Change as needed.
ALERT_AQI_THRESHOLD = 151

# -------------------------
# 1) Helper: Preprocess (re-use Module1)
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

print("Loading & preprocessing data...")
df_daily = preprocess(DATA_CSV)

if TARGET not in df_daily.columns:
    raise ValueError(f"Target {TARGET} not in data. Columns: {df_daily.columns.tolist()}")

series = df_daily[[TARGET]].dropna()
dates = series.index

# Train/test split consistent with Module2: last FORECAST_HORIZON days are test
train = series.iloc[:-FORECAST_HORIZON]
test = series.iloc[-FORECAST_HORIZON:]

print(f"Series length {len(series)} days. Train {len(train)} days, Test {len(test)} days.")

# -------------------------
# 2) AQI calculation (US EPA breakpoints for PM2.5, PM10)
# -------------------------
# Breakpoint tables (value_low, value_high, AQI_low, AQI_high)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500)
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500)
]

def compute_sub_aqi(conc, breakpoints):
    """
    Linear interpolation between breakpoints.
    """
    if np.isnan(conc):
        return np.nan
    for (c_low, c_high, a_low, a_high) in breakpoints:
        if c_low <= conc <= c_high:
            aqi = ( (a_high - a_low)/(c_high - c_low) ) * (conc - c_low) + a_low
            return round(aqi, 0)
    # conc > highest breakpoint -> cap to 500
    return 500.0

def aqi_for_row(row):
    """
    Compute AQI for row (expects PM2.5 and PM10 columns). Returns dict with sub-aqis and overall.
    """
    pm25 = row.get('PM2.5', np.nan)
    pm10 = row.get('PM10', np.nan)
    aqi_pm25 = compute_sub_aqi(pm25, PM25_BREAKPOINTS)
    aqi_pm10 = compute_sub_aqi(pm10, PM10_BREAKPOINTS)
    # overall AQI is max of pollutant AQIs considered
    overall = np.nanmax([aqi_pm25 if not np.isnan(aqi_pm25) else -np.inf,
                         aqi_pm10 if not np.isnan(aqi_pm10) else -np.inf])
    if overall == -np.inf:
        overall = np.nan
    return {
        'AQI_PM2.5': aqi_pm25,
        'AQI_PM10': aqi_pm10,
        'AQI': overall
    }

def aqi_category(aqi_val):
    if np.isnan(aqi_val):
        return "NA"
    aqi_val = float(aqi_val)
    if aqi_val <= 50:
        return "Good"
    if aqi_val <= 100:
        return "Moderate"
    if aqi_val <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi_val <= 200:
        return "Unhealthy"
    if aqi_val <= 300:
        return "Very Unhealthy"
    if aqi_val <= 500:
        return "Hazardous"
    return "Hazardous"

# -------------------------
# 3) Load models if present & forecast
# -------------------------
def find_model_files(model_dir, target):
    files = {}
    # ARIMA (pmdarima saved or statsmodels pickle)
    arima_candidates = glob.glob(os.path.join(model_dir, f"arima*{target}*.pkl"))
    if arima_candidates:
        files['ARIMA'] = arima_candidates[0]
    # Prophet
    prophet_candidates = glob.glob(os.path.join(model_dir, f"prophet*{target}*.pkl"))
    if prophet_candidates:
        files['Prophet'] = prophet_candidates[0]
    # LSTM (keras .keras or .h5)
    lstm_candidates = glob.glob(os.path.join(model_dir, f"lstm*{target}*.keras")) + glob.glob(os.path.join(model_dir, f"lstm*{target}*.h5"))
    if lstm_candidates:
        files['LSTM'] = lstm_candidates[0]
    # scaler (joblib)
    scaler_candidates = glob.glob(os.path.join(model_dir, f"scaler*{target}*.save"))
    if scaler_candidates:
        files['Scaler'] = scaler_candidates[0]
    return files

model_files = find_model_files(MODEL_DIR, TARGET)
print("Detected model files:", model_files)

forecasts = {}   # store pandas Series for each model

# Helper to safe pickle load
def safe_pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ARIMA forecasting
if 'ARIMA' in model_files:
    arima_path = model_files['ARIMA']
    print("Loading ARIMA from", arima_path)
    try:
        # try pmdarima load interface:
        try:
            from pmdarima.arima import ARIMA as PMDARIMA_ARIMA
            arima_obj = PMDARIMA_ARIMA.load(arima_path)
            preds = arima_obj.predict(n_periods=FORECAST_HORIZON)
            forecasts['ARIMA'] = pd.Series(preds, index=test.index)
        except Exception:
            # fallback: statsmodels pickle
            arima_obj = safe_pickle_load(arima_path)
            # statsmodels ARIMA fitted has forecast method
            preds = arima_obj.forecast(steps=FORECAST_HORIZON)
            forecasts['ARIMA'] = pd.Series(preds, index=test.index)
    except Exception as e:
        print("ARIMA load/forecast failed:", e)

# Prophet forecasting
if 'Prophet' in model_files:
    prophet_path = model_files['Prophet']
    print("Loading Prophet from", prophet_path)
    try:
        m = safe_pickle_load(prophet_path)
        # Prophet API: build future df with dates equal to the test index
        # make a dataframe starting from last train date for FORECAST_HORIZON days
        last_train_date = train.index[-1]
        future_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        fcst = m.predict(future_df)
        prophet_preds = fcst['yhat'].values
        forecasts['Prophet'] = pd.Series(prophet_preds, index=test.index)
    except Exception as e:
        print("Prophet load/forecast failed:", e)

# LSTM forecasting (iterative multi-step)
if 'LSTM' in model_files and 'Scaler' in model_files and TENSORFLOW_AVAILABLE and JOBLIB_AVAILABLE:
    lstm_path = model_files['LSTM']
    scaler_path = model_files['Scaler']
    print("Loading LSTM model:", lstm_path, "and scaler:", scaler_path)
    try:
        model_lstm = load_model(lstm_path)
        scaler = joblib.load(scaler_path)
        # prepare iterative forecasting:
        # take last LSTM_TIMESTEPS from series (scaled) and iteratively predict next value, append and shift
        # detect LSTM timesteps from model input shape
        timesteps = model_lstm.input_shape[1]
        arr = series[TARGET].values.reshape(-1,1)
        scaled_all = scaler.transform(arr)
        last_seq = scaled_all[-timesteps:].reshape(1, timesteps, 1)
        preds_scaled = []
        for _ in range(FORECAST_HORIZON):
            p = model_lstm.predict(last_seq, verbose=0)[0,0]
            preds_scaled.append(p)
            # update last_seq: append p and drop first
            new_seq = np.append(last_seq.flatten()[1:], p)
            last_seq = new_seq.reshape(1, timesteps, 1)
        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
        forecasts['LSTM'] = pd.Series(preds, index=test.index)
    except Exception as e:
        print("LSTM load/forecast failed:", e)
else:
    if 'LSTM' in model_files:
        print("LSTM model found but TensorFlow or joblib not available; skipping LSTM forecasting.")

# If no models found, quick fallback: naive persistence forecast (last observed value)
if not forecasts:
    print("No models detected; using persistence (last value) forecast as fallback.")
    last_val = series.iloc[-1,0]
    forecasts['Persistence'] = pd.Series([last_val]*FORECAST_HORIZON, index=test.index)

# -------------------------
# 4) Build a combined results dataframe: actual + model preds
# -------------------------
results_df = pd.DataFrame(index=test.index)
results_df['Actual'] = test[TARGET].values
for name, ser in forecasts.items():
    # align to test dates
    results_df[name] = ser.reindex(test.index).values

# Also attach full historical series for plotting long-term trends
full_df = df_daily.copy()
# compute AQI for historical full_df (using PM2.5 & PM10)
aqis = full_df.apply(lambda r: pd.Series(aqi_for_row(r)), axis=1)
full_df = pd.concat([full_df, aqis], axis=1)
full_df['AQI_category'] = full_df['AQI'].apply(aqi_category)

# For the forecasted days, compute AQI per model predictions (we'll compute for each model separately)
model_aqi_dfs = {}
for name in forecasts:
    temp = pd.DataFrame(index=test.index)
    temp[TARGET] = forecasts[name].values
    # try to preserve PM10 if available from other models? For now compute AQI using PM2.5 forecast and PM10=NaN
    # If you also forecast PM10 in your pipeline, compute similarly.
    temp['PM10'] = np.nan
    a = temp.apply(lambda r: pd.Series(aqi_for_row(r)), axis=1)
    temp = pd.concat([temp, a], axis=1)
    temp['AQI_category'] = temp['AQI'].apply(aqi_category)
    model_aqi_dfs[name] = temp

# Compute alerts: where AQI >= threshold for any model and for true AQI
alerts = []
# For true AQI (if available) compute for test period
true_aqi_rows = []
for idx in test.index:
    row = full_df.loc[idx] if idx in full_df.index else None
    if row is not None:
        true_aqi = row.get('AQI', np.nan)
        cat = aqi_category(true_aqi)
        true_aqi_rows.append({'date': idx, 'AQI': true_aqi, 'category': cat, 'source': 'Actual'})
        if not np.isnan(true_aqi) and true_aqi >= ALERT_AQI_THRESHOLD:
            alerts.append({'date': idx, 'aqi': true_aqi, 'category': cat, 'source': 'Actual'})
# For model predictions:
for name, mdf in model_aqi_dfs.items():
    for idx, row in mdf.iterrows():
        if not np.isnan(row['AQI']) and row['AQI'] >= ALERT_AQI_THRESHOLD:
            alerts.append({'date': idx, 'aqi': row['AQI'], 'category': row['AQI_category'], 'source': name})

alerts_df = pd.DataFrame(alerts).drop_duplicates(subset=['date','source']).sort_values('date')
alerts_csv = os.path.join(OUTPUT_DIR, "alerts_table.csv")
alerts_df.to_csv(alerts_csv, index=False)
print(f"Alerts saved -> {alerts_csv}")

# -------------------------
# 5) Visualizations
# -------------------------
sns.set(style="whitegrid", context="talk")

# (A) Plot: Actual vs Predictions (test window)
plt.figure(figsize=(12,5))
plt.plot(results_df.index, results_df['Actual'], marker='o', label='Actual')
for c in results_df.columns:
    if c != 'Actual':
        plt.plot(results_df.index, results_df[c], marker='x', linestyle='--', label=f'{c} pred')
plt.title(f"{TARGET} - Actual vs Predictions (test window)")
plt.xlabel("Date")
plt.ylabel(f"{TARGET} concentration")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "predictions_vs_actual_test.png"))
plt.show()

# (B) AQI time series for historical + forecast (use Actual AQI from full_df, plus model AQIs)
plt.figure(figsize=(12,5))
# historical AQI (last 60 days)
hist_slice = full_df['AQI'].last('60D')
plt.plot(hist_slice.index, hist_slice.values, label='Historical AQI (last 60d)', linewidth=2)
# overlay predicted AQI for each model
for name, mdf in model_aqi_dfs.items():
    plt.plot(mdf.index, mdf['AQI'], marker='x', linestyle='--', label=f"{name} AQI (pred)")
plt.axhline(ALERT_AQI_THRESHOLD, color='r', linestyle=':', label=f'Alert threshold (AQI={ALERT_AQI_THRESHOLD})')
plt.title("AQI: historical + predicted")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "aqi_historical_and_predicted.png"))
plt.show()

# (C) Bar: pollutant seasonal means (historical)
plt.figure(figsize=(8,5))
polls = ['PM2.5','PM10','NO2']
present = [p for p in polls if p in full_df.columns]
full_df[present].resample('M').mean().plot(kind='line', figsize=(10,5))
plt.title("Monthly average pollutant concentrations (historical)")
plt.xlabel("Month")
plt.ylabel("Concentration")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_pollutant_trends.png"))
plt.show()

# (D) Alerts summary (table)
if not alerts_df.empty:
    print("\n=== ALERTS (detected) ===")
    print(alerts_df.to_string(index=False))
else:
    print("\nNo alerts detected for the forecast window (AQI >= {}).".format(ALERT_AQI_THRESHOLD))

# Save results dataframe (actual + predictions)
results_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_test_window.csv"))
print("Predictions saved ->", os.path.join(OUTPUT_DIR, "predictions_test_window.csv"))

# Save model-aqi merged outputs
for name, mdf in model_aqi_dfs.items():
    mdf.to_csv(os.path.join(OUTPUT_DIR, f"model_aqi_{name}.csv"))
print(f"Model AQI outputs saved to {OUTPUT_DIR}")

print("\nMilestone 3 complete. Visual outputs & alerts are in the 'outputs/' folder.")
