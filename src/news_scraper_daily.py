# src/news_scraper_daily.py

from newspaper import Article
from googlesearch import search
import pandas as pd
from datetime import datetime
import time
import os

DATA_PATH = "data/articles_raw.csv"
query = "FSSAI food safety violation site:ndtv.com OR site:thehindu.com OR site:indiatimes.com"

# Search URLs
print("🔍 Searching Google News...")
urls = list(search(query, num_results=25))
print(f"🔗 Found {len(urls)} URLs\n")

# Load existing articles if available
if os.path.exists(DATA_PATH):
    existing_df = pd.read_csv(DATA_PATH)
    existing_urls = set(existing_df['url'].tolist())
else:
    existing_df = pd.DataFrame()
    existing_urls = set()

# Scrape new articles only
new_data = []

for url in urls:
    if url in existing_urls:
        print(f"⏩ Skipped (already exists): {url}")
        continue

    try:
        article = Article(url)
        article.download()
        article.parse()

        new_data.append({
            "title": article.title,
            "content": article.text,
            "url": url,
            "published": article.publish_date.strftime('%Y-%m-%d') if article.publish_date else "Unknown",
            "scraped_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        print(f"✅ Scraped: {article.title}")
        time.sleep(1)

    except Exception as e:
        print(f"⚠️ Failed to scrape {url}: {e}")

# Append new data
if new_data:
    df_new = pd.DataFrame(new_data)
    combined_df = pd.concat([existing_df, df_new], ignore_index=True)
    os.makedirs("data", exist_ok=True)
    combined_df.to_csv(DATA_PATH, index=False)
    print(f"\n✅ Added {len(new_data)} new article(s) to articles_raw.csv")
else:
    print("📭 No new articles found.")
