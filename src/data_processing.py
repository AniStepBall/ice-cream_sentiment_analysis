import pandas as pd
import re
from textblob import TextBlob

def loading_data():
    """
    A function that will load the raw data files [raw directory]
    Return: the csv flies
    """
    reviews = pd.read_csv("../data/raw/reviews.csv", encoding='utf-8')
    products = pd.read_csv("../data/raw/products.csv", encoding='utf-8')
    return reviews, products

def cleaning_text(text):
    """
    Function that cleans the review file - including HTML, extra spaces, etc
    Args
    """
    if pd.isna(text):
        return ""
    
    text = re.sub(r'<[^>]+>', '', str(text))
    text = ' '.join(text.split())
    text = text.lower()

    return text

def basic_cleaning(reviews_df):
    """
    Basic cleaning of the reviews dataframe
    """
    clean_df = reviews_df.copy()
    clean_df['clean_text'] = clean_df['text'].apply(cleaning_text)

    clean_df = clean_df[clean_df['clean_text'] != '']

    #df[col].method(value, inplace=True) instead
    #clean_df['stars'].fillna(clean_df['stars'].median(), inplace=True)
    
    #df[col] = df[col].method(value)
    clean_df['stars'] = clean_df['stars'].fillna(clean_df['stars'].median())

    print(f"Started with {len(reviews_df)} reviews")
    print(f"After cleaning: {len(clean_df)} reviews")

    return clean_df

if __name__ == "__main__":
    reviews, products = loading_data()
    clean_reviews = basic_cleaning(reviews)
    print("Cleaning complete")