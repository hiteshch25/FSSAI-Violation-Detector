# src/interpretability_shap.py

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# Load enriched article data
DATA_PATH = Path("data/articles_with_sentiment.csv")
df = pd.read_csv(DATA_PATH)

# ✅ Ensure 'published' is in datetime format
df['published'] = pd.to_datetime(df['published'], errors='coerce')
df = df.dropna(subset=['published'])  # Remove rows with invalid dates

# Set index for time-based operations
df.set_index("published", inplace=True)

# ✅ Feature Engineering
df['day_of_week'] = df.index.dayofweek
df['article_length'] = df['content'].str.len()

# Make sure 'polarity' column exists
features = df[['polarity', 'day_of_week', 'article_length']].dropna()

# ✅ Target: daily article count
daily_counts = df.resample('D').size().to_frame("article_count")
labels = daily_counts.reindex(df.index)['article_count'].fillna(0)
labels = labels.loc[features.index]

# ✅ Train model
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

# ✅ SHAP explanation
explainer = shap.Explainer(rf_model, X_test)
shap_values = explainer(X_test)

# ✅ Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)

# ✅ Save output
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/shap_summary_plot.png")
print("✅ SHAP summary plot saved as outputs/shap_summary_plot.png")
