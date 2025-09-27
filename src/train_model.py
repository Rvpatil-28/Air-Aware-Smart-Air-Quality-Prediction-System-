
import argparse
import os
import pandas as pd
from prophet import Prophet
import joblib
from src.utils import load_and_preprocess

def prepare_prophet_df(df, target_col):
    # Prophet expects columns ds (date) and y (value)
    df = df[[target_col]].reset_index().rename(columns={df.index.name or 'index':'ds', target_col:'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df[['ds','y']]

def train_prophet(csv, target_col, model_out):
    df = load_and_preprocess(csv, resample_rule='D')
    prophet_df = prepare_prophet_df(df, target_col)
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Saved model to {model_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--target', default='pm25', help='column name for target pollutant (lowercase)')
    parser.add_argument('--model_out', default='models/prophet_pm25.joblib')
    args = parser.parse_args()
    train_prophet(args.csv, args.target, args.model_out)
