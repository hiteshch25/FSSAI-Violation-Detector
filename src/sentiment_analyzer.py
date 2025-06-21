import pandas as pd
from textblob import TextBlob
from datetime import datetime

# Load scraped articles
df = pd.read_csv('data/articles_raw.csv')

# Rename columns to match forecasting input expectations
df.rename(columns={
    'text': 'content',
    'publish_date': 'published'
}, inplace=True)

# Function to analyze sentiment
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

# Apply sentiment analysis
df[['polarity', 'subjectivity']] = df['content'].apply(lambda x: pd.Series(get_sentiment(str(x))))

# Classify sentiment based on polarity
def classify_sentiment(p):
    if p > 0.1:
        return 'positive'
    elif p < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['polarity'].apply(classify_sentiment)
df['analyzed_at'] = datetime.now()

# Save updated file
df.to_csv('data/articles_with_sentiment.csv', index=False)
print("✅ Updated sentiment analysis saved to data/articles_with_sentiment.csv")
