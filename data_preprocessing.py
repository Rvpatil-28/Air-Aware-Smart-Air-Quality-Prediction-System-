
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Air_Quality.py/sample_air_quality.csv")
print("Dataset preview:\n", df.head())

df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)


df = df.interpolate(method='time')
df = df.bfill().ffill()   # ✅ fixed warning

# (2) Handle Outliers
for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3']:
    if col in df.columns:
        q_low = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        df[col] = np.clip(df[col], q_low, q_high)

# (3) Resample to daily averages
df_daily = df.resample('D').mean()

df_daily['day'] = df_daily.index.day
df_daily['month'] = df_daily.index.month
df_daily['year'] = df_daily.index.year
df_daily['dayofweek'] = df_daily.index.dayofweek

def season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df_daily['season'] = df_daily['month'].apply(season)

print("\n✅ Preprocessing complete! Processed dataset preview:\n", df_daily.head())


# 1. Pollutant trend over time
plt.figure(figsize=(12, 6))
for col in ['PM2.5', 'PM10', 'NO2']:
    if col in df_daily.columns:
        plt.plot(df_daily.index, df_daily[col], label=col)
plt.title("Daily Pollutant Trends")
plt.legend()
plt.show()

# 2. Correlation heatmap (only numeric columns)
plt.figure(figsize=(8, 6))
numeric_df = df_daily.select_dtypes(include=[np.number])  # ✅ fixes string issue
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation between Pollutants")
plt.show()

# 3. Seasonal pollutant averages
if 'season' in df_daily.columns:
    df_daily.groupby('season')[['PM2.5','PM10','NO2']].mean().plot(kind='bar', figsize=(8,6))
    plt.title("Seasonal Average Pollutant Levels")
    plt.ylabel("Concentration")
    plt.show()

