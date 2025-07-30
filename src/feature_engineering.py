import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

nlp = spacy.load("en_core_web_sm")
if 'spacytextblob' not in nlp.pipe_names:
    nlp.add_pipe("spacytextblob")

vader_analyzer = SentimentIntensityAnalyzer()

def extract_basic_features(text):
    """
    Extract simple features from text
    """
    if pd.isna(text) or text == '':
        return {
            'text_length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'capital_ratio': 0,
            'avg_word_length': 0
        }

    words = text.split()
    text_length = len(text)
    word_count = len(text.split())

    sentence = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentence if s.strip()])

    exclamation_count = text.count('!')
    question_count = text.count('?')

    capital_count = sum(1 for c in text if c.isupper())
    capital_ratio = capital_count / text_length if text_length >= 1 else 0

    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

    return {
        'text_length': text_length,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'capital_ratio': capital_ratio,
        'avg_word_length': avg_word_length
    }

def extract_sentiment_features(text):
    """Extracting sentimental features using spaCy with TextBlob"""
    if pd.isna(text) or text == '':
        return {'spacy_polarity': 0, 'spacy_subjectivity': 0, 'entities_count': 0}

    doc = nlp(text)
    return {
        'spacy_polarity': doc._.blob.polarity, 
        'spacy_subjectivity': doc._.blob.subjectivity, 
        'entities_count': len(doc.ents)
    }

def extract_vader_sentiment(text) -> dict:
    """
    Extracting sentiment using VADER (Valence Aware Dictionary sEntiment Reasoner)
    """
    if pd.isna(text) or text == '':
        return {'vader_compound': 0, 'vader_positive': 0, 'vader_negative': 0, 'vader_neutral': 0}

    scores = vader_analyzer.polarity_scores(text)
    return {
        'vader_compound': scores['compound'], 
        'vader_positive': scores['pos'], 
        'vader_negative': scores['neg'], 
        'vader_neutral': scores['neu']
    }

def extract_brands(merged_df):
    """
    Extract brand-related featured including popularity and average ratings
    Args:
        merged_df (pd.DataFrame): DataFrame with brand information
    Returns:
        pd.DataFrame: Dataframe with added brand features
    """
    brands_count = merged_df['brand_x'].value_counts()
    merged_df['brand_popularity'] = merged_df['brand_x'].map(brands_count)

    brand_average = merged_df.groupby('brand_x')['rating'].mean()
    merged_df['brand_average'] = merged_df['brand_x'].map(brand_average)

    return merged_df

def create_rating_category(star_review):
    """
    Creates a sentiment categories based on the individual review stars
    """
    if star_review >= 4:
        return 'positive'
    elif star_review <= 2:
        return 'negative'
    else:
        return 'neutral' 

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combining 3 different feature extract types in the dataframe
    Args:
        df(pd.DataFrame): the dataframe used to get all features
    Return:
        pd.DataFrame: combining all 3 features to the dataframe
    """

    extracting_type = [
        ("text", extract_basic_features),
        #("spacy", extract_sentiment_features),
        #("vader", extract_vader_sentiment)
    ]

    feature_dfs = []
    
    for i, j in extracting_type:
        print(f"Extracting {i} features...")
        extracted = df['clean_text'].apply(j)
        extracted_df = pd.DataFrame(extracted.tolist())
        feature_dfs.append(extracted_df)

    if 'brand_x' in df.columns:
        df = extract_brands(df)

    features_df = pd.concat([df] + feature_dfs, axis=1)
    
    print("clean_text sample:", features_df['clean_text'].iloc[0])
    print("Type:", type(features_df['clean_text'].iloc[0]))

    return features_df

if __name__ == "__main__":
    sample_text = "Super good, don't get me wrong. But I came for the caramel and brownies, not the sweet cream. \
        The packaging made it seem like brownies were packed and bountiful *crying frowny emoji* I'd say the taste \
        of this was amazing, but the ratio of brownie to sweet cream was disappointing. Liked it regardless but \
        probably won't buy again simply because it didn't live up to its promising package. I'll find another \
        one that has a better ratio and wayyy more yummy chewy brownies. Overall, good flavor, texture, idea, \
        and brownies. Not so great caramel/sweet cream/ brownie RATIO. Just add more brownies. Please."

    print("Text features: ", extract_basic_features(sample_text))
    print("spaCy features: ", extract_sentiment_features(sample_text))
    print("VADER features: ", extract_vader_sentiment(sample_text))
