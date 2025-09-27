from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta
from src.utils import load_and_preprocess, add_date_features, pm25_to_aqi

st.set_page_config(page_title='AirAware — AQ Forecast', layout='wide')

st.title('AirAware — Air Quality Forecast & Alerts (Demo)')

uploaded = st.file_uploader('Upload CSV (date column + pollutant columns like pm25, pm10, no2)', type=['csv'])
use_example = st.button('Use example dataset')

if uploaded is None and not use_example:
    st.info('Upload a CSV or click "Use example dataset" to load the supplied sample.')
    st.stop()

if uploaded:
    df = load_and_preprocess(uploaded)
else:
    df = load_and_preprocess('airaware_project/synthetic_air_quality_1yr.csv')

st.sidebar.header('Settings')
pollutant = st.sidebar.selectbox(
    'Select pollutant to forecast',
    options=[c for c in df.columns if c.startswith('pm') or c in ['pm25', 'pm10', 'no2']],
    index=0
)
horizon_days = st.sidebar.slider('Forecast horizon (days)', 7, 90, 14)

# --- Historical data ---
st.header('Historical data')
st.line_chart(df[pollutant])

# --- AQI gauge ---
st.header('Simple AQI gauge (latest)')
latest_val = df[pollutant].iloc[-1]
aqi_latest = pm25_to_aqi(latest_val) if 'pm25' in pollutant else int(np.nan)
st.metric(label=f'Latest {pollutant} (value)', value=f"{latest_val:.2f}")
st.metric(label='Estimated AQI (from PM2.5)', value=aqi_latest)

# --- Forecast with Prophet ---
st.header('Forecast (Prophet)')
st.write('The app includes code to train a Prophet model (see `src/train_model.py`). For demo speed we do a lightweight internal fit on the selected pollutant.')

from prophet import Prophet
prophet_df = df[[pollutant]].reset_index().rename(columns={df.index.name or 'index': 'ds', pollutant: 'y'})
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
m.fit(prophet_df)
future = m.make_future_dataframe(periods=horizon_days)
forecast = m.predict(future)
forecast_plot = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'yhat': 'Forecast'})
st.line_chart(forecast_plot)

# --- Alerting ---
st.header('Alerting — high AQI days (derived from PM2.5 forecast)')
if 'pm25' in df.columns:
    if pollutant != 'pm25':
        st.write('Note: AQI thresholds are estimated from PM2.5. Use PM2.5 as pollutant to get AQI-based alerts.')

    pm25_df = df[['pm25']].reset_index().rename(columns={df.index.name or 'index': 'ds', 'pm25': 'y'})
    m2 = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m2.fit(pm25_df)
    future2 = m2.make_future_dataframe(periods=horizon_days)
    fc2 = m2.predict(future2).set_index('ds')
    fc2['aqi'] = fc2['yhat'].apply(pm25_to_aqi)

    risky = fc2[fc2['aqi'] > 100]
    st.write(f"Predicted days with AQI > 100 in next {horizon_days} days: {len(risky)}")

    if not risky.empty:
        st.dataframe(risky[['yhat', 'aqi']].head(20))
    else:
        st.write('No high-AQI days predicted in the horizon.')

st.markdown('---')

# --- OpenAQ Fetch ---
st.sidebar.header('Real data (OpenAQ)')
use_openaq = st.sidebar.checkbox('Fetch real data from OpenAQ', value=False)
openaq_city = st.sidebar.text_input('City name for OpenAQ', value='Delhi')
openaq_days = st.sidebar.slider('Days to pull from OpenAQ', 7, 180, 60)

if use_openaq:
    st.info(f'Fetching {openaq_days} days of PM2.5 data for {openaq_city} from OpenAQ...')
    try:
        from src.fetch_openaq import fetch_openaq_city
        odf = fetch_openaq_city(openaq_city, parameter='pm25', days=openaq_days)
        if odf.empty:
            st.warning('OpenAQ returned no data for that city / parameter.')
        else:
            odf_daily = odf['value'].resample('D').mean().to_frame('pm25')
            st.success(f'Fetched {len(odf)} measurements. Aggregated to {len(odf_daily)} daily rows.')
            st.write('Sample of fetched data:')
            st.dataframe(odf_daily.tail(10))

            df = df.combine_first(odf_daily) if 'df' in locals() else odf_daily
    except Exception as e:
        st.error(f'Error fetching OpenAQ data: {e}')

# --- Model comparison: Prophet vs Rolling Mean ---
st.header('Model comparison (Prophet vs Rolling Mean baseline)')
if pollutant in df.columns:
    pf = df[[pollutant]].reset_index().rename(columns={df.index.name or 'index': 'ds', pollutant: 'y'})
    pf['ds'] = pd.to_datetime(pf['ds'])

    train = pf.iloc[:-14] if len(pf) > 30 else pf
    test = pf.iloc[-14:] if len(pf) > 14 else pf

    # Prophet fit
    try:
        m_cmp = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
        m_cmp.fit(train)
        future_cmp = m_cmp.make_future_dataframe(periods=len(test), freq='D')
        fc_cmp = m_cmp.predict(future_cmp).set_index('ds')
        prophet_pred = fc_cmp['yhat'].iloc[-len(test):].values
    except Exception as e:
        st.warning(f'Prophet model failed: {e}. Showing baseline only.')
        prophet_pred = None

    # Rolling mean baseline
    window = min(7, max(3, len(train) // 10))
    baseline_pred = train['y'].rolling(window=window).mean().iloc[-1]
    baseline_preds = np.full(shape=len(test), fill_value=float(baseline_pred))

    # Metrics
    from sklearn.metrics import mean_absolute_error

    def rmse(a, b):
        try:
            return mean_squared_error(a, b, squared=False)
        except TypeError:
            return np.sqrt(mean_squared_error(a, b))

    if prophet_pred is not None:
        p_rmse = rmse(test['y'].values, prophet_pred)
        p_mae = mean_absolute_error(test['y'].values, prophet_pred)
        st.write(f'Prophet — RMSE: {p_rmse:.2f}, MAE: {p_mae:.2f}')

    b_rmse = rmse(test['y'].values, baseline_preds)
    b_mae = mean_absolute_error(test['y'].values, baseline_preds)
    st.write(f'Rolling mean baseline (window={window}) — RMSE: {b_rmse:.2f}, MAE: {b_mae:.2f}')

    # Plot
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines+markers', name='Actual'))
    if prophet_pred is not None:
        fig.add_trace(go.Scatter(x=test['ds'], y=prophet_pred, mode='lines+markers', name='Prophet'))
    fig.add_trace(go.Scatter(x=test['ds'], y=baseline_preds, mode='lines+markers', name='RollingBaseline'))
    fig.update_layout(title='Test period: Actual vs Predictions', xaxis_title='Date', yaxis_title=pollutant)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.write('Selected pollutant not found in dataset for comparison.')

st.write('Project skeleton and training code included. Replace sample CSV with your own city/station CSV to run on real data.')
