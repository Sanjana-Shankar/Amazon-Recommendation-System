import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA

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
#feature_cols += [c for c in df.columns if c.startswith("bert_")]
feature_cols = [c for c in df.columns if c.startswith("bert_") and pd.api.types.is_numeric_dtype(df[c])]
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

print("\nGenerating predictions for entire dataset...")

# Predict for all rows in the dataset (not just test set)
all_predictions = pipeline.predict(X)
df["svm_predicted_label"] = all_predictions

# Standardize naming: convert camelCase → snake_case if present
rename_map = {
    "productId": "product_id",
    "userId": "user_id",
    "Id": "review_id"
}

df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)

# REQUIRED columns
required = {"user_id", "product_id"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns for evaluation: {missing}")

print("Column validation passed!")

# Create evaluation dataframe
svm_df = df[["user_id", "product_id"]].copy()
svm_df.rename(columns={"product_id": "item_id"}, inplace=True)
svm_df["score"] = all_predictions

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "svm_predictions.csv")
svm_df.to_csv(OUTPUT_CSV, index=False)

print(f"[OK] SVM predictions written to: {OUTPUT_CSV}")

# Create output dataframe
'''
output_df = df[id_cols + ["svm_predicted_label"]] if id_cols else df[["svm_predicted_label"]]

# Save to CSV
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "svm_predictions.csv")
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"SVM predictions saved to: {OUTPUT_CSV}")
'''

# ---- Score distribution on all data ----
if hasattr(pipeline.named_steps["svm"], "decision_function"):
    all_scores = pipeline.decision_function(X)
    plt.figure(figsize=(6, 4))
    plt.hist(all_scores, bins=50)
    plt.xlabel("Decision score")
    plt.ylabel("Count")
    plt.title("Distribution of SVM Decision Scores (All Data)")
    plt.tight_layout()
    plt.savefig("./eval_outputs/svm_score_distribution.png", dpi=150)
    plt.show()
    print("Score distribution plot saved to ./eval_outputs/svm_score_distribution.png")

# ---- Confusion Matrix Plot ----
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(cmap="Blues", ax=ax, colorbar=False)
ax.set_title("SVM Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("./eval_outputs/svm_confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix plot saved to ./eval_outputs/svm_confusion_matrix.png")

# ---- ROC Curve Plot (using decision_function) ----
if hasattr(pipeline.named_steps["svm"], "decision_function"):
    # Scores for the positive class
    y_score = pipeline.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"LinearSVC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SVM ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("./eval_outputs/svm_roc_curve.png", dpi=150)
    plt.show()
    print("ROC curve plot saved to ./eval_outputs/svm_roc_curve.png")
else:
    print("ROC curve not available: SVM model has no decision_function.")
