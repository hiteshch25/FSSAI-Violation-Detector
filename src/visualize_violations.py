# src/visualize_violations.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load flagged violations
df = pd.read_csv("data/articles_flagged.csv", parse_dates=["published", "flagged_at"])

# Ensure 'published' is datetime for resampling
df['published'] = pd.to_datetime(df['published'], errors='coerce')
df = df.dropna(subset=['published'])

# Plot 1: Time trend of violations (by published date)
daily_counts = df.resample('D', on='published').size()

plt.figure(figsize=(12, 5))
daily_counts.plot(marker='o')
plt.title("Daily FSSAI Violations Flagged")
plt.xlabel("Date Published")
plt.ylabel("Number of Violations")
plt.grid(True)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/violations_trend.png")
print("✅ Saved: outputs/violations_trend.png")

# Plot 2: Violations by day of the week
df['day_of_week'] = df['published'].dt.day_name()
weekday_counts = df['day_of_week'].value_counts().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

plt.figure(figsize=(10, 5))
weekday_counts.plot(kind='bar', color='tomato')
plt.title("Violations by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Number of Violations")
plt.tight_layout()
plt.savefig("outputs/violations_by_weekday.png")
print("✅ Saved: outputs/violations_by_weekday.png")

# Optional Plot 3: By Source (if source column exists)
if 'source' in df.columns:
    plt.figure(figsize=(10, 5))
    df['source'].value_counts().plot(kind='barh', color='skyblue')
    plt.title("Violations by Source")
    plt.xlabel("Number of Violations")
    plt.ylabel("News Source")
    plt.tight_layout()
    plt.savefig("outputs/violations_by_source.png")
    print("✅ Saved: outputs/violations_by_source.png")

print("Visualization complete!")
