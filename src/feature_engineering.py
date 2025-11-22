import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pickle

vader = SentimentIntensityAnalyzer()

# Load BERT model once globally (faster
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

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
    
def add_sentiment_features(df):
    
    print("Computing VADER sentiment...")

    df["sentiment_review"] = df["reviewText"].fillna("").apply(
        lambda t: vader.polarity_scores(t)["compound"]
    )

    df["sentiment_summary"] = df["summary"].fillna("").apply(
        lambda t: vader.polarity_scores(t)["compound"]
    )
    return df

# Text Features from Reviews turn them into encodings for review input (MiniLM)
def compute_miniLM_embedding(text_list, batch_size=32):
    # Return the MiniLM embedding for large batches keeping consistent outputs while improving the speed
    print("Generating fast MiniLM embeddings...")
    embeddings = model.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")

def add_miniLM_features(df, batch_size=32):
    # Adds miniLM embeddings to the dataframe, using batched encoding
    
    texts = df["reviewText"].fillna("").tolist()

    print("Running batched MiniLM embedding generation...")
    
    embeddings = compute_miniLM_embedding(df["reviewText"].tolist())
    print("Final MiniLM embedding shape:", embeddings.shape)
    np.save("../data/review_embeddings.npy", embeddings)
    print("Saved embeddings to ../data/review_embeddings.npy")
    
    # Add a column indicating the embedding file
    df["bert_embedding_file"] = "../data/review_embeddings.npy"
    return df

# Category Encoding 
def add_category_features(df):
    if "rating" not in df.columns:
        print("No rating column found — skipping one-hot encoding.")
        return df
    
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    print(f"Encoder {enc}")
    rating_encoded = enc.fit_transform(df[["rating"]])
    print(f"Rating encoded {rating_encoded}")
    
    rating_df = pd.DataFrame(
        rating_encoded,
        columns=enc.get_feature_names_out(["rating"])
    )
    print(f"Rating df: {rating_df}")
    
    df = pd.concat([df.reset_index(drop=True), rating_df], axis=1)
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
    df = add_miniLM_features(df)

    df = add_category_features(df)
    df = normalize_rating(df)

    print('Finished feature engineering! ')
    return df


path = "../data/Reviews.csv"
df = feature_engineering_pipeline(path)
df.to_pickle("../data/feature_engineered_reviews.pkl")

with open("../data/feature_engineered_reviews.pkl", "rb") as f:
    obj = pickle.load(f)

print(obj)


