
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

AQI_BREAKPOINTS_PM25 = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

def pm25_to_aqi(conc):
    """Convert PM2.5 concentration (µg/m3) to US EPA-like AQI using linear interpolation across breakpoints."""
    try:
        c = float(conc)
    except:
        return np.nan
    for (c_low, c_high, i_low, i_high) in AQI_BREAKPOINTS_PM25:
        if c_low <= c <= c_high:
            aqi = (i_high - i_low)*(c - c_low)/(c_high - c_low) + i_low
            return round(aqi)
    return 500

def load_and_preprocess(csv_path, date_col_candidates=['date','ds','timestamp'], resample_rule='D'):
    df = pd.read_csv(csv_path)
    # find date column
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    else:
        # fallback: first column
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # forward fill small gaps and then resample
    df = df.resample(resample_rule).mean()
    df = df.interpolate(limit=3).ffill().bfill()
    # standard column names: lowercase and remove spaces
    df.columns = [c.strip().lower().replace(' ','').replace('.','') for c in df.columns]
    return df

def add_date_features(df):
    d = df.copy()
    d['dayofweek'] = d.index.dayofweek
    d['month'] = d.index.month
    d['day'] = d.index.day
    d['is_weekend'] = d['dayofweek'].isin([5,6]).astype(int)
    return d

def scale_series(series):
    scaler = MinMaxScaler()
    arr = series.values.reshape(-1,1)
    scaled = scaler.fit_transform(arr).flatten()
    return scaled, scaler
