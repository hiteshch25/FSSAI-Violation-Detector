# src/load_forecasting_ann.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os

# Load data
DATA_PATH = Path("data/articles_with_sentiment.csv")
df = pd.read_csv(DATA_PATH)

# Parse and clean date
df['published'] = pd.to_datetime(df['published'], errors='coerce')
df = df.dropna(subset=['published'])
df.set_index('published', inplace=True)

# Resample to daily article counts
daily_counts = df.resample('D').size().to_frame("article_count")
daily_counts = daily_counts.asfreq('D').fillna(0)

# Step 1: Debug check
print("✅ Available days of data:", len(daily_counts))
print("📅 Date range:", daily_counts.index.min().date(), "to", daily_counts.index.max().date())
print("📊 Last 10 days:\n", daily_counts.tail(10))

# Normalize values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_counts)

# Create X and y sequences for ANN
def create_sequences(data, window=7):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, window=7)

# Step 2: Handle insufficient data
if len(X) == 0:
    raise ValueError("❌ Not enough data to create training sequences. You need at least 8 days of article data.")

X = X.reshape(X.shape[0], X.shape[1], 1)

# Build ANN model
model = Sequential([
    Flatten(input_shape=(X.shape[1], 1)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=8, verbose=1)

# Forecast next 10 days
last_sequence = scaled_data[-7:]
predictions = []

for _ in range(10):
    input_seq = last_sequence[-7:].reshape(1, 7, 1)
    next_val = model.predict(input_seq, verbose=0)[0][0]
    predictions.append(next_val)
    last_sequence = np.append(last_sequence, [[next_val]], axis=0)

# Inverse scale the predictions
forecasted_counts = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Create forecast DataFrame
forecast_dates = pd.date_range(start=daily_counts.index[-1] + pd.Timedelta(days=1), periods=10)
forecast_df = pd.DataFrame({"date": forecast_dates, "forecasted_articles": forecasted_counts})

# Save forecast output
os.makedirs("data", exist_ok=True)
forecast_df.to_csv("data/daily_forecast.csv", index=False)

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(daily_counts[-30:].index, daily_counts[-30:].values, label="Last 30 days")
plt.plot(forecast_df['date'], forecast_df['forecasted_articles'], label="Forecast (next 10)", linestyle='--')
plt.xlabel("Date")
plt.ylabel("Article Count")
plt.title("📰 Daily Food Article Count Forecast")
plt.legend()
plt.grid()
plt.tight_layout()

# Save plot and model
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
plt.savefig("outputs/forecast_graph.png")
model.save("models/ann_model.h5")

print("\n✅ Forecast complete!")
print("📁 Saved: data/daily_forecast.csv")
print("📈 Saved: outputs/forecast_graph.png")
print("🧠 Saved: models/ann_model.h5")
