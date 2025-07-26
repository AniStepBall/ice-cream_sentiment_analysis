# Ice Cream Review Sentiment System

A project that analyzes, models, serves customer sentiment for ice cream reviews using NLP and machine learning.

## Project Goal
Build a sentiment analysis pipeline using real-world ice cream review data. The goal is to explore, clean, and extract insights and features from the data to prepare it for machine learning.

## Learning Goals
- Structuring a real-world data science project
- Cleaning and exploring the datasets (reviews.csv and product.csv)
- Baseline models, comparisions, optimizing the model and create documentation
- With RESTful API with model serving and model versioning with MLflow

## Project Structure
```
ice-cream_sentiment_analysis/
├── data/
│ ├── raw/
│ └── processed/
├── models/
│ ├── final_sentiment_model.pkl
│ └── model_documentation.json
├── notebooks/
│ ├── 01_explore_data.ipynb
│ ├── 02_data_exploration.ipynb
│ ├── 03_feature_engineering.ipynb
│ └── 04_model_training.ipynb
├── src/
│ ├── data_processing.py
│ ├── features.py
│ └── models.py
├── api/
│ └── app.py
├── mlruns/
├── requirements.txt
└── README.md
```
## Steup Instructions

### 1. Clone the repo and set up environment

```bash
git clone <ice-cream_sentiment_analysis>
cd ice-cream_sentiment_analysis
python -m venv .venv
source .venv/bin/activate   #.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```


## Next Tasks
- Adding machine learning models