import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModel
import torch

# Load BERT model once globally (faster
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

# Feature engineering pipeline 

# Data loading  
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    print("Columns in dataset:", df.columns.tolist())
    return df

# Basic cleaning 
def clean_dataframe(df):
    # Normalize column names (lowercase and consistent with our pipeline)
    df = df.rename(columns={
        "ProductId": "productId",
        "UserId": "userId",
        "Score": "rating",
        "Text": "reviewText",
        "Summary": "summary",
        "Time": "timestamp"
    })

    print("Columns after renaming:", df.columns.tolist())

    df = df.dropna(subset=["reviewText", "rating", "productId", "userId"])

    # Convert timestamp 
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    # Drop rows with bad timestamps 
    df = df.dropna(subset=["timestamp"])

    return df


# User-Product Interaction Features 
def add_user_product_features(df):
    # average rating per product 
    df["avg_rating_per_product"] = df.groupby("productId")["rating"].transform("mean")
    print("Average rating per product")

    # number of reviews per suer 
    df["num_reviews_per_user"] = df.groupby("userId")["rating"].transform("count")
    print("Number reviews per user")

    # recency / frequency (convert timestamp -> datetime)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    newest_time = df["timestamp"].max()
    print(f"Newest time {newest_time}")

    # recency score = how recent the review is (lower = older)
    df["recency_days"] = (newest_time - df["timestamp"]).dt.days
    print("Recency days")

    ninety_days_ago = newest_time - pd.Timedelta(days=90)
    print(f"Ninety days ago {ninety_days_ago}")

    df["recent_user_frequency"] = (
        df.groupby("userId")["timestamp"]
        .transform(lambda x: (x > ninety_days_ago).sum())
    )

    return df

# Sentiment Features (VADER)

def sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    try:
        score = analyzer.polarity_scores(text)["compound"]
        print(f"Sentiment score: {score}")
        return score
    except:
        print("Sentiment score: 0.0")
        return 0.0
    
def add_sentiment_features(df):
    df["sentiment_review"] = df["reviewText"].apply(sentiment_score)
    print("Sentiment review")
    df["sentiment_summary"] = df["summary"].apply(sentiment_score)
    print("Sentiment summary")
    return df

# Text Features from Reviews turn them into encodings for review input (BERT)
def compute_bert_embedding(text):
    # Return the 768-dim BERT embedding for one text input
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(768)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128, # keeps runtime reasonable
        padding="max_length"
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # CLS token embedding = outputs.last_hidden_state[:,0,:]
    cls_embedding = outputs.last_hidden_state[:,0,:].squeeze().numpy()
    return cls_embedding 

def add_bert_features(df):
    # 768-dimensional BERT embeddings 
    # Adds columns: bert_0 ..... bert 767
    print("Generating BERT embeddings for reviewText...")

    bert_embeddings = df["reviewText"].apply(compute_bert_embedding)

    # Convert list of arrays -> 2D matrix 
    bert_matrix = np.vstack(bert_embeddings.values)

    bert_df = pd.DataFrame(
        bert_matrix,
        columns = [f"bert_{i}" for i in range (bert_matrix.shape[1])]
    )

    print("BERT embeddings shape:", bert_df.shape)

    df = pd.concat([df.reset_index(drop=True), bert_df], axis=1)
    return df

'''
def add_tfidf_features(df, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    print(f"Vectorizer {vectorizer}")
    tfidf_matrix = vectorizer.fit_transform(df["reviewText"])
    print(f"TF-IDF matrix {tfidf_matrix}")

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    )
    print(f"TF-IDF {tfidf_df}")
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    return df
'''

# Category Encoding 
def add_category_features(df):
    if "category" not in df.columns:
        print("No category column found - skipping one-hot encoding.")
        return df
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    print(f"Encoder {enc}")
    category_encoded = enc.fit_transform(df[["category"]])
    print(f"Category encoded {category_encoded}")

    cat_df = pd.DataFrame(category_encoded, columns=enc.get_feature_names_out(["category"]))
    print(f"Cat df: {cat_df}")
    df = pd.concat([df.reset_index(drop=True), cat_df], axis=1)
    print(f"df: {df}")
    return df

# Rating Normalization 
def normalize_rating(df):
    df["normalized_rating"] = (df["rating"]-1) / 4
    print("Normalized rating")
    return df

# Final Function Outputting a Clean DataFrame

def feature_engineering_pipeline(path):
    df = load_data(path)
    df = clean_dataframe(df)

    df = add_user_product_features(df)
    df = add_sentiment_features(df)

    # BERT features
    df = add_bert_features(df)

    df = add_category_features(df)
    df = normalize_rating(df)

    print('Finished feature engineering! ')
    return df


path = "../data/Reviews.csv"
df = feature_engineering_pipeline(path)
df.to_pickle("..data/feature_engineered_reviews.pkl")


