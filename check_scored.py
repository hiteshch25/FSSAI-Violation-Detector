import pandas as pd

df = pd.read_csv('data/articles_scored.csv')
print("📰 Total Articles Scraped:", len(df))
print("\n🧠 Sample Sentiments:\n", df[['text', 'sentiment', 'polarity']].head())
