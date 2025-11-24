# ncf_model.py
import os
import pickle
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import time

# Neural Collaborative Filtering style recommender 

# CONFIG 
FEATURE_PKL = "../data/feature_engineered_reviews.pkl" # From feature_engineering pipeline
EMBEDDING_NPY = "../data/review_embeddings.npy" # From feature_engineering pipeline
MODEL_OUT = "../models/ncf_model.pth"
ENCODERS_OUT = "../models/encoders.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM_USER = 64
EMBED_DIM_ITEM = 64
HIDDEN_DIMS = [128, 64]
BATCH_SIZE = 1024
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-6
TOP_K = 10
RANDOM_SEED = 42 

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Loads feature-engineered DataFrame and MiniLM embeddings (path ffrom pipeline)
def load_feature_engineered_data(feature_pkl_path: str) -> pd.DataFrame:
    if not os.path.exists(feature_pkl_path):
        raise FileNotFoundError(f"Feature file not found: {feature_pkl_path}")
    with open(feature_pkl_path, "rb") as f:
        df = pickle.load(f)

    #Ensure expected columns exist
    required_cols = ["userId", "productId", "rating", "reviewText", "bert_embedding_file"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' in feature file")
    return df.reset_index(drop=True)

def load_embeddings(npy_path: str) -> np.ndarray:
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Embeddings file not found: {npy_path}")
    emb = np.load(npy_path)
    return emb.astype("float32")

# Builds user/item id mappings
# Creates a PyTorch dataset that includes user, item, sentiment, and review_embedding 
class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray, user_encoder: LabelEncoder, item_encoder: LabelEncoder):
        '''
        df: Feature-engineered DataFrame (rows correspond to embeddings order)
        embeddings: numpy array shape (n_rows, emb_dim)
        Encoders map original IDs to integer indices
        '''
        assert len(df) == len(embeddings), "Dataframe length must match embeddings length"

        self.df = df.copy().reset_index(drop=True)
        self.embeddings = embeddings
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder 

        # Binary label: positive if rating >=4 else 0
        self.df["label"] = (self.df["rating"] >= 4).astype("float32")

        # Sentiment if present or zeros 
        if "sentiment_review" not in self.df.columns:
            self.df["sentiment_review"] = 0.0

        # Precompute encoded indices
        self.user_idx = user_encoder.transform(self.df["userId"].astype(str))
        self.item_idx = item_encoder.transform(self.df["productId"].astype(str))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        u = int(self.user_idx[idx])
        i = int(self.item_idx[idx])
        emb = self.embeddings[idx] # float32 vector
        sentiment = float(self.df.loc[idx, "sentiment_review"])
        label = float(self.df.loc[idx, "label"])

        return {
            "user": u,
            "item": i,
            "emb": emb,
            "sentiment": sentiment,
            "label": label
        }

### NCF MODEL ### 
# Trains an NCF-style model that learns user and item embeddings and combines them with dense layers 
class NCFModel(nn.Module):
    def __init__(self,
                 n_users: int,
                 n_items: int,
                 emb_dim_user: int,
                 emb_dim_item: int,
                 review_emb_dim: int,
                 hidden_dims: List[int]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim_user)
        self.item_emb = nn.Embedding(n_items, emb_dim_item)

        # small MLP for review embeddings + sentiment 
        review_in = review_emb_dim + 1
        self.review_mlp = nn.Sequential(
            nn.Linear(review_in, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        # combine user/item embeddings + review features
        review_in = emb_dim_user + emb_dim_user + hidden_dims[1]
        self.comb_mlp = nn.Sequential(
            nn.Linear(review_in, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1) # logit
        )
    
    def forward(self, user_idx, item_idx, review_emb, sentiment):
        u = self.user_emb(user_idx) # (B, emb_dim_user)
        v = self.item_emb(item_idx) # (B, emb_dim_item)
        
        x = torch.cat([review_emb, sentiment.unsqueeze(1)], dim=1)
        r = self.review_mlp(x)
        combined = torch.cat([u, v, r], dim=1)
        out = self.comb_mlp(combined).squeeze(1) # (B,)
        return out 
    
### TRAIN / EVAL HELPERS ####
def collate_fn(batch):
    # batch is list of dicts
    users = torch.tensor([b["user"] for b in batch], dtype=torch.long)
    items = torch.tensor([b["item"] for b in batch], dtype=torch.long)
    embs = torch.from_numpy(np.array([b["emb"] for b in batch], dtype=np.float32))
    sentiments = torch.tensor([b["sentiment"] for b in batch], dtype=torch.float32)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return users, items, embs, sentiments, labels

# Uses binary target (positive if rating >= 4) and BCEWithLogitsLoss
def train_one_epoch(model, loader, opt, criterion):
    model.train()
    running_loss = 0.0
    for users, items, embs, sentiments, labels in loader:
        users = users.to(DEVICE)
        items = items.to(DEVICE)
        embs = embs.to(DEVICE)
        sentiments = sentiments.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(users, items, embs, sentiments)
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item() * users.size(0) 
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds = []
    truths = []
    for users, items, embs, sentiments, labels in loader:
        users = users.to(DEVICE)
        items = items.to(DEVICE)
        embs = embs.to(DEVICE)
        sentiments = sentiments.to(DEVICE)
        logits = model(users, items, embs, sentiments)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
        truths.append(labels.numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    auc = roc_auc_score(truths, preds) if len(np.unique(truths)) > 1 else float("nan")
    pred_labels = (preds >= 0.5).astype(int)
    acc = accuracy_score(truths, pred_labels)
    return {"auc": auc, "accuracy": acc, "preds": preds, "truths": truths}

# Evalutes accuracy/AUC and produces per-user top-K recommendations 
@torch.no_grad() 
def recommend_for_user(model:nn.Module,
                       user_encoded: int,
                       all_item_indices: np.ndarray,
                       itemid_by_index: Dict[int, str],
                       user_interacted_items: set,
                       review_embs_for_user_placeholder: np.ndarray,
                       sentiment_placeholder: float,
                       top_k: int = 10) -> List[Tuple[str, float]]:
    '''
    Produces top-K item recommendations for a user by scoring all items. Because review embedding 
    is user-review specific, we pass a placeholder review embedding. 
    '''
    model.eval()
    n_items = len(all_item_indices)
    batch_size = 2048
    scores = []
    device = DEVICE

    # Pre-build user batch base tensor
    user_idx_tensor = torch.tensor([user_encoded], dtype=torch.long, device=device)

    for start in range(0, n_items, batch_size):
        end = min(n_items, start + batch_size)

        # Correct item batch for this slice
        item_batch = torch.arange(start, end, dtype=torch.long, device=device)

        # User batch repeated for this batch size
        user_batch = user_idx_tensor.repeat(end - start)

        # Review embedding repeated for batch
        emb_batch = torch.tensor(
            np.repeat(review_embs_for_user_placeholder[np.newaxis, :], end - start, axis=0),
            dtype=torch.float32,
            device=device
        )

        # Sentiment repeated for batch
        sentiment_batch = torch.tensor(
            np.repeat([sentiment_placeholder], end - start),
            dtype=torch.float32,
            device=device
        )

        # This is where to add with torch.no_grad()
        with torch.no_grad():
            logits = model(user_batch, item_batch, emb_batch, sentiment_batch)

        probs = torch.sigmoid(logits).cpu().numpy()
        scores.append(probs)

    # Correct concatenation
    scores = np.concatenate(scores)

    # Mask items user already interacted with 
    ranked_idx = np.argsort(-scores)
    results = []
    for idx in ranked_idx:
        item_index = all_item_indices[idx]
        if item_index in user_interacted_items:
            continue
        itemid = itemid_by_index[int(item_index)]
        results.append((itemid, float(scores[idx])))
        if len(results) >= top_k:
            break
    return results

def main():
    print("Loading feature-engineered data from:", FEATURE_PKL)
    df = load_feature_engineered_data(FEATURE_PKL)

    # Load embeddings file referenced by df (we assume same ordering)
    print("Loading embeddings from:", EMBEDDING_NPY)
    embeddings = load_embeddings(EMBEDDING_NPY)  # shape (n_rows, emb_dim)
    review_emb_dim = embeddings.shape[1]

    # --- FIX #1: ensure df and embeddings align ---
    print("DF rows:", len(df), "Embeddings rows:", embeddings.shape[0])
    if embeddings.shape[0] != len(df):
        if embeddings.shape[0] > len(df):
            print("Trimming embeddings to match df rows.")
            embeddings = embeddings[: len(df), :]
        else:
            raise RuntimeError(
                f"Embeddings shorter ({embeddings.shape[0]}) than df ({len(df)}). "
                "Regenerate embeddings using the final df ordering."
            )

    # Create label encoders for users and items
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df["userId"] = df["userId"].astype(str)
    df["productId"] = df["productId"].astype(str)
    user_encoder.fit(df["userId"])
    item_encoder.fit(df["productId"])

    n_users = len(user_encoder.classes_)
    #n_items = len(item_encoder.classes_)
    n_items = df["productId"].nunique()
    print(f"n_users={n_users}, n_items={n_items}, review_emb_dim={review_emb_dim}")

    # --- FIX #2: validate encoded item indices ---
    item_indices = item_encoder.transform(df["productId"])
    if item_indices.max() >= n_items:
        raise RuntimeError(
            f"Item encoder produced index {item_indices.max()} >= n_items ({n_items}). "
            "Encoder/data mismatch."
        )

    dataset = ReviewDataset(df, embeddings, user_encoder, item_encoder)

    # Train/val split
    val_frac = 0.1
    n_val = int(len(dataset) * val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = NCFModel(n_users=n_users,
                     n_items=n_items,
                     emb_dim_user=EMBED_DIM_USER,
                     emb_dim_item=EMBED_DIM_ITEM,
                     review_emb_dim=review_emb_dim,
                     hidden_dims=HIDDEN_DIMS).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, opt, criterion)
        metrics = evaluate(model, val_loader)
        val_auc = metrics["auc"]
        val_acc = metrics["accuracy"]
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f} | val_acc={val_acc:.4f}")
        # Save best
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            # Save encoders
            with open(ENCODERS_OUT, "wb") as f:
                pickle.dump({
                    "user_encoder": user_encoder,
                    "item_encoder": item_encoder
                }, f)
            print(f"Saved best model to {MODEL_OUT} (val_auc={val_auc:.4f})")

    print("Finished training loop. now checkpoing handling...")
    print("MODEL_OUT exists?", os.path.exists(MODEL_OUT))
    # right before loading best model
    print("About to load model from", MODEL_OUT)
    model_load_start = time.time()
    # Load best model for recommendations
    if os.path.exists(MODEL_OUT):
        model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    print("Loaded model in", time.time() - model_load_start, "seconds")
    
    # Precompute mapping for recommendations
    all_item_indices = np.arange(n_items)
    # right before itemid_by_index creation
    print("About to build itemid_by_index (this can be slow). n_items =", n_items)
    t0 = time.time()
    itemid_by_index = {i: item for i, item in enumerate(item_encoder.inverse_transform(np.arange(n_items)))}
    print("Built itemid_by_index in", time.time() - t0, "seconds")

    # right before building user_to_items
    print("About to build user_to_items mapping (iterating df rows)...")
    t1 = time.time()
    # build user_to_items...
    print("Built user_to_items in", time.time() - t1, "seconds")

    # Convert IDs once (much faster than encoding in a loop)
    df["u_enc"] = user_encoder.transform(df["userId"].astype(str))
    df["i_enc"] = item_encoder.transform(df["productId"].astype(str))

    # Build mapping: user_encoded_id -> set(item_encoded_ids)
    user_to_items = (
        df.groupby("u_enc")["i_enc"]
        .apply(lambda s: set(s.values))
        .to_dict()
    )

    print("Built user_to_items in", time.time() - t1, "seconds")

    # Build user -> set(items) interactions to filter out already seen
    
    # Example: produce top-K recommendations for first 5 users
    '''
    print("Generating sample recommendations for first 5 users...")
    sample_users = list(range(min(5, n_users)))
    for u_enc in sample_users:
        # For placeholder review embedding & sentiment, use user-average of their reviews if available
        #user_rows = df[df["userId"] == user_encoder.inverse_transform([u_enc])[0]]
        orig_id = user_encoder.classes_[u_enc]
        user_rows = df.loc[df["userId"] == orig_id]
        if len(user_rows) > 0:
            idxs = user_rows.index.tolist()
            user_emb_placeholder = np.mean(embeddings[idxs], axis=0)
            sentiment_placeholder = float(user_rows["sentiment_review"].mean())
        else:
            # cold-start: global average
            user_emb_placeholder = embeddings.mean(axis=0)
            sentiment_placeholder = float(df["sentiment_review"].mean())

        recs = recommend_for_user(
            model=model,
            user_encoded=int(u_enc),
            all_item_indices=all_item_indices,
            itemid_by_index=itemid_by_index,
            user_interacted_items=user_to_items.get(int(u_enc), set()),
            review_embs_for_user_placeholder=user_emb_placeholder,
            sentiment_placeholder=sentiment_placeholder,
            top_k=TOP_K
        )
        print(f"User {user_encoder.inverse_transform([u_enc])[0]} top-{TOP_K} recommendations:")
        for pid, score in recs:
            print(f"  {pid} (score={score:.4f})")
        print("-" * 30)
    '''
    # -------------------------------------------------------------
    # Generate recommendations for EVERY user and save to CSV
    # (ONE ROW PER USER)
    # -------------------------------------------------------------
    print("Generating grouped recommendations for ALL users...")

    user_rows_out = []  # one entry per user
    output_path = "../data/all_user_recommendations_grouped.csv"

    for u_enc in range(n_users):

        orig_user_id = user_encoder.classes_[u_enc]

        # Compute placeholder embedding & sentiment
        user_rows = df.loc[df["userId"] == orig_user_id]
        if len(user_rows) > 0:
            idxs = user_rows.index.tolist()
            user_emb_placeholder = np.mean(embeddings[idxs], axis=0)
            sentiment_placeholder = float(user_rows["sentiment_review"].mean())
        else:
            user_emb_placeholder = embeddings.mean(axis=0)
            sentiment_placeholder = float(df["sentiment_review"].mean())

        # Get top-K recommendations
        recs = recommend_for_user(
            model=model,
            user_encoded=int(u_enc),
            all_item_indices=all_item_indices,
            itemid_by_index=itemid_by_index,
            user_interacted_items=user_to_items.get(int(u_enc), set()),
            review_embs_for_user_placeholder=user_emb_placeholder,
            sentiment_placeholder=sentiment_placeholder,
            top_k=TOP_K
        )

        # Separate product IDs and scores into lists
        product_ids = [pid for pid, _ in recs]
        scores = [score for _, score in recs]

        # Save ONE ROW per user
        user_rows_out.append({
            "user_id": orig_user_id,
            "user_encoded": u_enc,
            "recommended_product_ids": ",".join(product_ids),
            "scores": ",".join([f"{s:.6f}" for s in scores])
        })

        # progress
        if (u_enc + 1) % 10000 == 0:
            print(f"Processed {u_enc + 1}/{n_users} users...")

    # Save as DataFrame
    output_df = pd.DataFrame(user_rows_out)
    output_df.to_csv(output_path, index=False)

    print(f"\nSaved grouped recommendations to: {output_path}")
    print("Total users saved:", len(output_df))

    print("Done.")

if __name__ == "__main__":
    main()

# Saves the trained model and encoders 



