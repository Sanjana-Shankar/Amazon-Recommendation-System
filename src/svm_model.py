import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# CONFIG
PICKLE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feature_engineered_reviews.pkl")   # Path to feature-engineered DataFrame
MODEL_OUT = os.path.join(os.path.dirname(__file__), "models", "svm_model.joblib")
TARGET = "binary"   # "binary" or "multiclass"
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# Load feature-engineered DataFrame 
if not os.path.exists(PICKLE_PATH):
    raise FileNotFoundError(f"Feature-engineered file not found at {PICKLE_PATH}. Run the feature engineering pipeline and save the dataframe to this path.")
df = pd.read_pickle(PICKLE_PATH)

# Select features: all BERT dims + numeric metadata + one-hot category columns (if present)
feature_cols = []
feature_cols += [c for c in df.columns if c.startswith("bert_")]
feature_cols += [c for c in ["sentiment_review", "sentiment_summary",
                             "avg_rating_per_product", "num_reviews_per_user",
                             "recency_days", "recent_user_frequency"] if c in df.columns]
feature_cols += [c for c in df.columns if c.startswith("category_")]

# Check: Make sure features exist
if len(feature_cols) == 0:
    raise ValueError("No feature columns found. Ensure BERT embeddings and other features exist in the dataframe.")

X = df[feature_cols].values

# Prepare target
if TARGET == "binary":
    # Binary: positive if rating >=4
    if "rating" not in df.columns:
        raise ValueError("rating column missing for binary target.")
    y = (df["rating"] >= 4).astype(int).values
else:
    # Multiclass: original 1-5 rating (can be used for groupings: 1-2 (negative), 3 (neutral), 4-5 (positive))
    if "rating" not in df.columns:
        raise ValueError("rating column missing for multiclass target.")
    y = df["rating"].astype(int).values

# Train / test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                    random_state=RANDOM_STATE,
                                                    stratify=y)     # stratify to maintain class balance

# Two types of SVM pipelines to try
# Use a pipeline: scaler + SVM.
use_linear = True  # switch to False to try RBF kernel SVC
if use_linear:
    # Linear SVM pipeline (linear boundary): good for larger datasets and high-dimensional data
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(class_weight="balanced", max_iter=10000, random_state=RANDOM_STATE))
    ])
else:
    # RBF kernel SVM pipeline (non-linear boundary): 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=RANDOM_STATE))
    ])

print("Training SVM...")
pipeline.fit(X_train, y_train)

# Evaluation: Accuracy, classification report, confusion matrix
print("Evaluating...")
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")