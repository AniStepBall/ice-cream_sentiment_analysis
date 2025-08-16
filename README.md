# Ice Cream Review Sentiment System

A project that analyzes, models, and determines  customer sentiment for ice cream reviews using NLP and machine learning.

This repository contains a **sentiment analysis framework** that classifies product reviews into **positive, neutral, or negative sentiment**.  
The project emphasizes:
- Clean data preprocessing
- Rich feature engineering (linguistic + sentiment + brand-level features)
- Multiple baseline models with evaluation
- Model optimization and hyperparameter tuning
- Interpretability and visualization

Unlike typical sentiment classifiers, this project:
•	Integrates multiple feature extraction techniques (basic text stats, spaCy sentiment, VADER sentiment, brand-level metadata).
•	Trains and evaluates several baseline machine learning models (Logistic Regression, Naive Bayes, Linear SVM, Gradient Boosting, and Voting Classifiers).
•	Performs cross-validation and hyperparameter tuning to find optimal configurations.
•	Provides visualisation and interpretation tools (confusion matrices, heatmaps, feature importance).
•	Supports comparisons across models for accuracy, F1 score, training/inference speed, and robustness.


## Project Goal
Build a sentiment analysis pipeline using real-world ice cream review data. The goal is to explore, clean, and extract insights and features from the data to prepare it for machine learning.
This project is designed for an ice cream company aiming to better understand how customers perceive its products, packaging, and overall brand experience. While the technical focus is on building and optimizing sentiment analysis models, the ultimate business value lies in generating actionable insights from customer reviews.

Some key business problems this can help resolve include:
- Understand Customer Perception
- Identify how customers talk about flavors, texture, packaging, pricing, and overall satisfaction.
- Detect positive drivers (e.g., “rich flavor,” “creamy texture”) and negative pain points (e.g., “too sweet,” “poor packaging,” “not enough brownies”).
- Improve Product Offerings
- Use sentiment insights to refine product recipes (e.g., flavor balance, ingredient quality).
- Prioritize features that customers value most.
- Enhance Customer Experience
- Spot recurring issues (e.g., “ice cream melts quickly,” “packaging misleading”) to improve quality control.
- Adjust communication and branding to better align with customer expectations.
- Guide Marketing & Branding Strategies
- Tailor marketing messages around positively perceived attributes (e.g., “creamy & indulgent” vs. “low-calorie & refreshing”).
- Benchmark sentiment across product lines or brands to identify competitive strengths.
- Support Data-Driven Decision Making
- Provide management with dashboards and reports summarizing sentiment by brand, product, or time period.
- Enable early detection of negative trends, preventing churn and improving brand loyalty.

## Project Structure
```
ice-cream_sentiment_analysis/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── 01_explore_data.ipynb
│ ├── 02_data_exploration.ipynb
│ ├── 03_feature_engineering.ipynb
│ └── 04_model_training.ipynb
├── src/
│ ├── data_processing.py
│ ├── features.py
│ ├── analyzer.py
│ ├── optimizer.py
│ ├── trainer.py
│ └── models.py
├── requirements.txt
└── README.md
```
**key features **
- **Data Processing**
  - Cleans raw text (removes HTML, lowercasing, spacing).
  - Filters reviews with missing or empty text.
  - Derives sentiment labels from star ratings.

- **Feature Engineering**
  - Text features: length, word count, punctuation, capitalization.
  - Sentiment features: spaCy+TextBlob polarity/subjectivity, VADER compound score.
  - Brand features: popularity & average ratings.

- **Model Training**
  - Baseline models: Logistic Regression, Naive Bayes, Linear SVM, Gradient Boosting, Voting Ensemble.
  - TF-IDF vectorization with up to 5,000 features.
  - Stratified K-Fold Cross Validation.
  - Performance metrics: Accuracy, Weighted F1, Training/Inference times.

- **Model Analysis**
  - Compare models side-by-side (accuracy, F1, time).
  - Confusion matrix for best model.
  - Heatmaps for performance metrics.
  - Feature importance ranking for interpretability.

- **Optimization**
  - GridSearchCV for hyperparameter tuning.
  - Supports multiple models (Logistic Regression, NB, SVM, Gradient Boosting).
  - Testing optimized models on real-world example reviews.
 
## Example Results
- **Baseline Logistic Regression**: ~80% accuracy, strong F1 performance.
- **Optimized Gradient Boosting**: Improved F1 with tradeoff in inference time.
- **Model Comparison**: Voting Ensemble performed competitively across multiple metrics.

Visualizations include:  
✅ Confusion matrix  
✅ Model performance bar chart  
✅ Feature importance (TF-IDF terms, sentiment drivers)  
✅ Performance heatmap  


## Step Instructions

### 1. Clone the repo and set up the environment

```bash
git clone <ice-cream_sentiment_analysis>
cd ice-cream_sentiment_analysis
python -m venv .venv
source .venv/bin/activate   #.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
**Setup & Usage**
1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
Install dependencies:

```bash

pip install -r requirements.txt
Preprocess and engineer features:

```bash

python scripts/data_processing.py
python scripts/feature_engineering.py
Train and evaluate models:

```bash
python scripts/models.py

**Future Enhancements**
Integrate deep learning models (BERT, RoBERTa, DistilBERT).
Expand feature engineering with word embeddings (GloVe, fastText).
Add explainability with SHAP/LIME for model transparency.
Improve class balancing with SMOTE or focal loss.
Deploy with Streamlit for interactive review sentiment demo.
Set up MLflow or Weights & Biases for experiment tracking.
Support multilingual sentiment analysis with HuggingFace transformers.

**References**
SpaCy: https://spacy.io
VaderSentiment: https://github.com/cjhutto/vaderSentiment
scikit-learn: https://scikit-learn.org/stable/

**Next Tasks**
- Adding API features

