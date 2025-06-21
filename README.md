# FSSAI Food Safety Violation Detector

This project automates the detection, forecasting, and explanation of FSSAI-related food safety violations reported in Indian news sources. It uses Python, data science techniques, and a Streamlit dashboard to provide an end-to-end pipeline from scraping to interpretability.

## Project Features (Phase 1–6)

1. **News Scraping**
   - Dynamically scrapes food safety-related articles using Google Search and newspaper3k
   - Stores unique articles in `data/articles_raw.csv`

2. **Sentiment Analysis**
   - Uses TextBlob to analyze sentiment polarity and subjectivity of articles
   - Outputs to `data/articles_with_sentiment.csv`

3. **Violation Filtering**
   - Flags articles containing predefined FSSAI violation keywords
   - Stores flagged articles in `data/articles_flagged.csv`

4. **Forecasting with ANN**
   - Trains a neural network to predict future violation article volume
   - Outputs forecast to `data/daily_forecast.csv`

5. **Model Interpretability with SHAP**
   - Uses SHAP to interpret which features affect article volume
   - Saves SHAP plots in `outputs/`

6. **Streamlit Dashboard**
   - Displays scraped articles, sentiments, violations, forecasts, and SHAP explanations

---

## Folder Structure

```
FSSAI-Violation-Detector/
├── app.py
├── run_all.py
├── requirements.txt
├── data/
│   ├── articles_raw.csv
│   ├── articles_with_sentiment.csv
│   ├── articles_flagged.csv
│   └── daily_forecast.csv
├── outputs/
│   ├── shap_summary_plot.png
│   ├── violations_trend.png
│   └── violations_by_weekday.png
├── src/
│   ├── news_scraper_daily.py
│   ├── sentiment_analyzer.py
│   ├── violation_filter.py
│   ├── load_forecasting_ann.py
│   ├── interpretability_shap.py
│   └── visualize_violations.py
```

---

## How to Set Up and Run Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/FSSAI-Violation-Detector.git
cd FSSAI-Violation-Detector
```

### Step 2: Set Up Environment and Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# OR
source venv/bin/activate   # On macOS/Linux

pip install -r requirements.txt
```

### Step 3: Run the Full Pipeline
```bash
python run_all.py
```

### Step 4: Launch the Dashboard
```bash
streamlit run app.py
```
