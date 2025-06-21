import pandas as pd 
from datetime import datetime

# ✅ Load the new file with sentiment and content
df = pd.read_csv('data/articles_with_sentiment.csv')

# Define FSSAI violation-related keywords
violation_keywords = [
    "food safety", "spoiled", "rotten", "unhygienic", "contaminated",
    "expired", "FSSAI notice", "FSSAI violation", "unsafe food", "sealed", "raided"
]

# Check if any keyword exists in the article text
def contains_violation(text):
    text_lower = str(text).lower()
    return any(keyword in text_lower for keyword in violation_keywords)

# Flag and filter violations
df['is_flagged'] = df['content'].apply(contains_violation)
flagged_df = df[df['is_flagged']].copy()
flagged_df['flagged_at'] = datetime.now()

# Save results
flagged_df.to_csv('data/articles_flagged.csv', index=False)
print(f"🚨 {len(flagged_df)} violation(s) flagged and saved to data/articles_flagged.csv")
