# run_all.py

import os
import subprocess

def run_script(name, file_path):
    print(f"\n🚀 Running: {name}")
    try:
        subprocess.run(["python", file_path], check=True)
        print(f"✅ {name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} failed.")
        exit(1)

# Step 1: Scrape latest articles
run_script("Step 1: Scraping articles", "src/news_scraper_daily.py")

# Step 2: Sentiment analysis
run_script("Step 2: Sentiment scoring", "src/sentiment_analyzer.py")

# Step 3: Violation flagging
run_script("Step 3: Violation filter", "src/violation_filter.py")

# Step 4: Forecast article counts
run_script("Step 4: ANN forecasting", "src/load_forecasting_ann.py")

# Step 5: SHAP interpretability
run_script("Step 5: SHAP interpretability", "src/interpretability_shap.py")

# Step 6: Visualization prep (plots)
run_script("Step 6: Violation visualization", "src/visualize_violations.py")

print("\n🎉 All steps completed! You can now run the Streamlit app:")
print("👉 streamlit run app.py")
