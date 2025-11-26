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
from bert_encoder import run_bert_embeddings
import os

vader = SentimentIntensityAnalyzer()

# Load BERT model once globally (faster
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# bert_model = AutoModel.from_pretrained("bert-base-uncased")

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
    
def add_sentiment_features(df, cache_path="../data/vader_sentiment.npy"):
    
    print("Computing or loading VADER sentiment...")

    if os.path.exists(cache_path):
        print(f"Found existing VADER sentiment cache at {cache_path}. Loading file instead of computing...")
        try:
            cached_sentiment = np.load(cache_path, mmap_mode="r")
            if cached_sentiment.shape[0] == len(df) and cached_sentiment.shape[1] == 2:
                print("Loaded existing VADER sentiment successfully:", cached_sentiment.shape)
                df["sentiment_review"] = cached_sentiment[:, 0]
                df["sentiment_summary"] = cached_sentiment[:, 1]
                return df
            else:
                print("Shape mismatch — recomputing VADER sentiment.")
        except Exception as e:
            print(f"Failed to load existing VADER sentiment due to: {e}. Recomputing sentiment...")
    
    # If VADER sentiment not loaded or is invalid, compute it
    print("Calculating VADER sentiment scores...")
    review_sent = df["reviewText"].fillna("").apply(
        lambda t: vader.polarity_scores(t)["compound"]
    ).to_numpy()

    summary_sent = df["summary"].fillna("").apply(
        lambda t: vader.polarity_scores(t)["compound"]
    ).to_numpy()

    # Save cache as .npy
    sentiment_arr = np.vstack([review_sent, summary_sent]).T.astype("float32")  # shape: (n_rows, 2)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, sentiment_arr)
    print(f"Saved VADER sentiment cache to {cache_path} with shape {sentiment_arr.shape}")
    
    # Add to dataframe
    df["sentiment_review"] = review_sent
    df["sentiment_summary"] = summary_sent
    
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
    embeddings_path = "../data/review_embeddings.npy"

    # Check if embeddings file exists
    if os.path.exists(embeddings_path):
        print(f"Found existing MiniLM embeddings at {embeddings_path}. Loading file instead of computing...")
        try:
            # if exists, load file
            embeddings = np.load(embeddings_path, mmap_mode="r")
            print("Loaded existing embeddings successfully.")
            if embeddings.shape[0] != len(df):
                print("Warning: Loaded embeddings shape does not match dataframe length. Recomputing embeddings.")
                raise ValueError("Shape mismatch")

        except Exception as e:
            print(f"Failed to load existing embeddings due to: {e}. Recomputing embeddings...")
            embeddings = None
    else:
        embeddings = None

    # If embeddings not loaded, compute them
    if embeddings is None:
        print("Running batched MiniLM embedding generation...")
        texts = df["reviewText"].fillna("").tolist()
        embeddings = compute_miniLM_embedding(texts)
        print("Final MiniLM embedding shape:", embeddings.shape)
        np.save("../data/review_embeddings.npy", embeddings)
        print("Saved embeddings to ../data/review_embeddings.npy")
    
    # Add a column indicating the embedding file
    df["miniLM_embedding_file"] = embeddings_path
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

    try:
        print("Running BERT embeddings...")
        # keep original df intact, run BERT encoder which returns a mapping dataframe
        df = run_bert_embeddings(
            df,
            text_col="reviewText",
            out_path="../data/review_embeddings_bert.npy",
            model_name="bert-base-uncased",
            chunk_size=5000,
            batch_size=64,
            pooling="mean",
            normalize=True,
            overwrite=False,
            save_mapping=True,
        )

    except Exception as e:
        print(f"Warning: run_bert_embeddings failed: {e} -- continuing without BERT embeddings")

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


