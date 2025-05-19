import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import pickle
from datetime import datetime

# Load dataset
try:
    df = pd.read_csv('healthcare_monitoring_dataset.csv')
except FileNotFoundError:
    print("Error: 'healthcare_monitoring_dataset.csv' not found. Ensure the file exists.")
    exit(1)

# Convert Timestamp to datetime and set as index
try:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
except KeyError:
    print("Error: 'Timestamp' column not found or invalid format.")
    exit(1)

# Ensure Heart Rate is numeric
try:
    df['Heart Rate'] = pd.to_numeric(df['Heart Rate'], errors='coerce')
except KeyError:
    print("Error: 'Heart Rate' column not found.")
    exit(1)

# Filter for a single patient (e.g., PAT1003)
patient_id = 'PAT1003'
patient_data = df[df['Patient ID'] == patient_id]
if patient_data.empty:
    print(f"Error: No data found for Patient ID '{patient_id}'. Available IDs: {df['Patient ID'].unique()[:5]}...")
    exit(1)

# Select Heart Rate for time series analysis
heart_rate = patient_data['Heart Rate'].dropna()
print(f"Initial Heart Rate data points for {patient_id}: {len(heart_rate)}")

# Ensure data is evenly spaced (resample to 1-minute intervals, filling missing values)
try:
    heart_rate = heart_rate.resample('1min').mean().interpolate()
except ValueError as e:
    print(f"Error during resampling: {e}")
    exit(1)

print(f"Resampled Heart Rate data points: {len(heart_rate)}")

# Validate data for stationarity test
if len(heart_rate) < 10:
    print("Warning: Too few data points (< 10) for reliable ARIMA analysis. Using default differencing (d=0).")
    d = 0
elif heart_rate.nunique() <= 1:
    print("Warning: Heart Rate series is constant or has no variation. Using default differencing (d=0).")
    d = 0
else:
    # Check stationarity with Augmented Dickey-Fuller test
    def check_stationarity(series):
        try:
            result = adfuller(series, autolag='AIC')
            p_value = result[1]
            print(f"ADF p-value: {p_value:.4f}")
            return p_value < 0.05  # Stationary if p-value < 0.05
        except Exception as e:
            print(f"Error in ADF test: {e}. Assuming non-stationary with d=1.")
            return False

    # Differencing if non-stationary
    if not check_stationarity(heart_rate):
        heart_rate_diff = heart_rate.diff().dropna()
        if len(heart_rate_diff) > 10 and heart_rate_diff.nunique() > 1 and check_stationarity(heart_rate_diff):
            heart_rate = heart_rate_diff
            d = 1
        else:
            d = 2
    else:
        d = 0

# Split data: 80% train, 20% test
if len(heart_rate) < 20:
    print("Error: Insufficient data points (< 20) after preprocessing. Need more data for training and testing.")
    exit(1)

train_size = int(len(heart_rate) * 0.8)
train, test = heart_rate[:train_size], heart_rate[train_size:]

# Fit ARIMA model (order=(p,d,q), using 5,0,1 as a starting point)
try:
    model = ARIMA(train, order=(5, d, 1))
    model_fit = model.fit()
except Exception as e:
    print(f"Error fitting ARIMA model: {e}. Trying simpler order (1,0,1).")
    model = ARIMA(train, order=(1, 0, 1))
    model_fit = model.fit()

# Forecast for test period
forecast = model_fit.forecast(steps=len(test))

# Evaluate model
mae = mean_absolute_error(test, forecast)
print(f'Mean Absolute Error: {mae:.2f} BPM')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Data', color='blue')
plt.plot(test.index, test, label='Actual Heart Rate', color='green')
plt.plot(test.index, forecast, label='Forecasted Heart Rate', color='red', linestyle='--')
plt.title(f'ARIMA Forecast of Heart Rate for {patient_id}')
plt.xlabel('Timestamp')
plt.ylabel('Heart Rate (BPM)')
plt.legend()
plt.grid(True)
plt.savefig('arima_heart_rate_forecast.png')
plt.close()

# Save model for real-time use
with open('arima_model.pkl', 'wb') as f:
    pickle.dump(model_fit, f)

# Real-time forecasting (next 10 minutes)
future_steps = 10
future_forecast = model_fit.forecast(steps=future_steps)
future_index = pd.date_range(start=heart_rate.index[-1], periods=future_steps+1, freq='1min')[1:]
future_df = pd.DataFrame({'Forecasted Heart Rate': future_forecast}, index=future_index)
print('\nFuture 10-Minute Forecast:')
print(future_df)

# Insights
print('\nInsights from ARIMA Analysis:')
print(f'1. Model Fit: The ARIMA(5,{d},1) model achieved an MAE of {mae:.2f} BPM, indicating {"good" if mae < 5 else "moderate"} accuracy.')
print(f'2. Stationarity: The Heart Rate series required {d} order differencing, suggesting {"stable" if d == 0 else "trending"} behavior.')
print('3. Real-Time Potential: The model can predict Heart Rate for the next 10 minutes, enabling anomaly detection (e.g., Heart Rate > 100 BPM).')
print('4. Data Sparsity: Limited data points for a single patient may reduce model robustness. Consider aggregating across similar patients.')
print('5. Scalability: The saved model supports continuous monitoring with periodic updates.')