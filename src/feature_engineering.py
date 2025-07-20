import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
if 'spacytextblob' not in nlp.pipe_names:
    nlp.add_pipe("spacytextblob")

vader_analyzer = SentimentIntensityAnalyzer()

def extracting_feature(text):
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
            'capital_ratio': 0
        }
    
    text_length = len(text)
    word_count = len(text.split())
    sentence_count = len(text.split('.'))
    exclamation_count = text.count('!')
    question_count = text.count('?')

    capital_count = sum(1 for c in text if c.isupper())
    capital_ratio = capital_count / text_length if text_length > 0 else 0

    return {
        'text_length': text_length,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'capital_ratio': capital_ratio
    }

def extracting_sentiment_features(text):
    """Extracting sentimental features using spaCy with TextBlob"""
    if pd.isna(text) or text == '':
        return {'spacy_polarity': 0, 'spacy_subjectivity': 0, 'entities_count': 0}

    try:
        #text = str(text)[:1000]
        doc = nlp(text)
        return {
            'spacy_polarity': doc._.blob.polarity, 
            'spacy_subjectivity': doc._.blob.subjectivity, 
            'entities_count': len(doc.ents)
        }
    except Exception as e:
        print(f"spaCy error: {e}")
        return {'spacy_polarity': 0, 'spacy_subjectivity': 0, 'entities_count': 0}

def extracting_vader_sentiment(text):
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

def extracting_brands(merged_df):
    """"""
    brands_count = merged_df['brand_x'].value_counts()
    merged_df['brand_popularity'] = merged_df['brand_x'].map(brands_count)

    brand_average = merged_df.groupby('brand_x')['rating'].mean()
    merged_df['brand_average'] = merged_df['brand_x'].map(brand_average)

    return merged_df

def creating_rating(rating):
    """"""
    if rating >= 4:
        return 'positive'
    elif rating >= 3:
        return 'neutral'
    else:
        return 'negative' 

def brand_full_name(brand):
    """"""
    if brand == 'bj':
        return "Ben & Jerry's"
    elif brand == 'hd':
        return 'Häagen-Dazs'
    elif brand == 'breyers':
        return 'Breyers'
    else:
        return 'Talenti'

def creating_features(df):
    """
    Creating features for dataframe
    """
    print("Extracting text features...")
    text_features = df['clean_text'].apply(extracting_feature)
    text_features_df = pd.DataFrame(text_features.tolist())

    print("Extracting Spacy features...")
    spacy_features = df['clean_text'].map(extracting_sentiment_features)
    spacy_features_df = pd.DataFrame(spacy_features.tolist())

    print("Extracting VADER features...")
    vader_features = df['clean_text'].map(extracting_vader_sentiment)
    vader_features_df = pd.DataFrame(vader_features.tolist())

    features_df = pd.concat([df, text_features_df, spacy_features_df, vader_features_df], axis=1)

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

    print("Text features: ", extracting_feature(sample_text))
    print("scapy features: ", extracting_sentiment_features(sample_text))
    print("VADER features: ", extracting_vader_sentiment(sample_text))
