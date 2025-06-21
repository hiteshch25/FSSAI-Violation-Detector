import pandas as pd

df = pd.read_csv('data/articles_flagged.csv')
print("🔍 Rows in CSV:", len(df))
print("📅 Date column sample:\n", df['flagged_at'].head())
