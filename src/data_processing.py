import pandas as pd
import re

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function that will load the raw data files [raw directory]
    Return:
        tuple: (reviews.csv, products.csv)
    """
    reviews = pd.read_csv("../data/raw/reviews.csv", encoding='utf-8')
    products = pd.read_csv("../data/raw/products.csv", encoding='utf-8')
    return reviews, products

def clean_text(text: str) -> str:
    """
    Function that cleans the review file - including HTML, extra spaces, etc
    Args:
        text (str): Raw text to be cleaned
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    text = re.sub(r'<[^>]+>', '', str(text))
    text = ' '.join(text.split())
    text = text.lower()

    return text

def clean_basic_data(reviews_df:pd.DataFrame) -> pd.DataFrame:
    """
    Function that does a basic cleaning of the reviews csv file
    Args:
        reviews_df (pd.Dataframe): a dataframe of the reviews.csv file
    Returns:
        clean_df (pd.Dataframe): a copy of the review.csv file with basic cleaning perfomed
    """
    clean_df = reviews_df.copy()
    clean_df['clean_text'] = clean_df['text'].apply(clean_text)

    clean_df = clean_df[clean_df['clean_text'] != '']
    
    print(f"Reviews with missing stars: {clean_df['stars'].isna().sum()}")
    #clean_df['stars'] = clean_df['stars'].fillna(clean_df['stars'].median())
    
    clean_df = clean_df.dropna(subset=['stars'])

    print(f"Started with {len(reviews_df)} reviews")
    print(f"After cleaning: {len(clean_df)} reviews")

    return clean_df

if __name__ == "__main__":
    reviews, products = load_data()
    clean_reviews = clean_basic_data(reviews)
    print("Cleaning complete")