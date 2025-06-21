import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="FSSAI Violation Detector", layout="wide")

st.title("🔍 FSSAI Food Safety Violation Dashboard")

# --- Load Data Safely ---
try:
    sentiment_df = pd.read_csv("data/articles_with_sentiment.csv")
    flagged_df = pd.read_csv("data/articles_flagged.csv")
    forecast_df = pd.read_csv("data/daily_forecast.csv")
except FileNotFoundError as e:
    st.error(f"❌ Required file not found: {e.filename}")
    st.stop()

# --- Section 1: Scraped Articles Preview ---
st.header("📰 Scraped Articles (Sample)")
st.dataframe(sentiment_df[['title', 'content']].head(5), use_container_width=True)

# --- Section 2: Sentiment Overview ---
st.header("💬 Sentiment Summary")
sentiment_counts = sentiment_df['sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# --- Section 3: Violation Highlights ---
st.header("🚨 Flagged Violations")
st.dataframe(flagged_df[['title', 'content', 'flagged_at']].head(5), use_container_width=True)

# --- Section 4: Violation Trend (Bar Chart) ---
st.header("📅 Violation Trend by Date")

flagged_df['flagged_at'] = pd.to_datetime(flagged_df['flagged_at'], errors='coerce')
daily_violations = flagged_df['flagged_at'].dt.date.value_counts().sort_index()

fig1, ax1 = plt.subplots(figsize=(10, 4))
daily_violations.plot(kind='bar', color='tomato', ax=ax1)
ax1.set_title("Daily Count of FSSAI Violations")
ax1.set_xlabel("Date")
ax1.set_ylabel("Violations")
plt.xticks(rotation=45)
st.pyplot(fig1)

# --- Section 5: Forecast Plot ---
st.header("📈 Forecasted Articles (Next 10 Days)")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(forecast_df['date'], forecast_df['forecasted_articles'], marker='o', linestyle='--', color='green')
ax2.set_title("Forecasted Article Volume")
ax2.set_xlabel("Date")
ax2.set_ylabel("Predicted Articles")
plt.xticks(rotation=45)
st.pyplot(fig2)

# --- Section 6: SHAP Summary Plot ---
st.header("🧠 Feature Impact (SHAP)")

if os.path.exists("outputs/shap_summary_plot.png"):
    st.image("outputs/shap_summary_plot.png", caption="SHAP Summary Plot", use_column_width=True)
else:
    st.warning("SHAP summary plot not found. Please run interpretability_shap.py")
