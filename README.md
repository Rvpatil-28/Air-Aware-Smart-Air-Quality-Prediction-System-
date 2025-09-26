
# AirAware — Air Quality Prediction System (Demo)

This project contains a simple Air Quality Prediction system with:
- Data preprocessing utilities
- Prophet-based time-series training script
- Streamlit dashboard (`app.py`) to visualize historical data, forecasts and alerts
- Example synthetic dataset at `data/sample_aq.csv`

## How to run (quick)
1. Create a Python virtualenv and install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Train the model (optional for demo):
   ```
   python src/train_model.py --csv data/sample_aq.csv --target pm25 --model_out models/prophet_pm25.joblib
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

If you want to use your own CSV, provide a file with a date/time column (named `date` or `ds`) and pollutant columns like `pm25`, `pm10`, `no2`. The code is written to be flexible—see `src/utils.py` for details.


## Fetching real data from OpenAQ

Use `src/fetch_openaq.py` to pull PM2.5 or other pollutant data for a city. The Streamlit app can also fetch OpenAQ data when you enable it in the sidebar.

## Email Alerts

To enable email alerts, provide SMTP settings and use `src/alerts.py`. This is a placeholder that demonstrates sending an alert email; configure carefully for your SMTP provider.
