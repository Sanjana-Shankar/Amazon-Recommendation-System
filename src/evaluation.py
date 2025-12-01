# v# evaluation.py  (UPDATED for hybrid evaluation)
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging

# Try UMAP, but fall back to TSNE if not available
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

# -----------------------------
# Basic metrics
# -----------------------------
def rmse(y_true, y_pred):
    """Compute RMSE between arrays or pandas series."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def precision_recall_f1_at_k(recs_by_user, relevant_by_user, K):
    """
    Compute Precision@K, Recall@K, F1@K averaged across users (macro-average).
    - recs_by_user: dict user_id -> list of top-K item_ids (ordered or unordered)
    - relevant_by_user: dict user_id -> set of relevant item_ids
    Returns tuple (precision, recall, f1).
    """
    precisions = []
    recalls = []
    f1s = []
    users = sorted(set(list(recs_by_user.keys()) + list(relevant_by_user.keys())))
    for u in users:
        recs = recs_by_user.get(u, [])[:K]
        rel = relevant_by_user.get(u, set())
        if len(recs) == 0 and len(rel) == 0:
            # skip users with no recs and no ground-truth relevant items
            continue
        hits = len([i for i in recs if i in rel])
        prec = hits / K
        rec = hits / len(rel) if len(rel) > 0 else 0.0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    if len(precisions) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))

# -----------------------------
# Helpers: build top-K recommendations from model scores
# -----------------------------
def topk_from_scores_df(scores_df, K):
    """
    scores_df: DataFrame with columns ['user_id', 'item_id', 'score'] (one row per user-item)
    returns dict: user_id -> list of top-K item_ids (sorted by descending score)
    """
    topk = {}
    grouped = scores_df.groupby('user_id')
    for uid, g in grouped:
        top_items = g.sort_values('score', ascending=False).head(K)['item_id'].tolist()
        topk[uid] = top_items
    return topk

def topk_from_prediction_df(pred_df, K):
    """
    pred_df: DataFrame with columns ['user_id', 'item_id', 'pred_rating'] (used for rating-based ranking)
    """
    df = pred_df.rename(columns={'pred_rating': 'score'})
    return topk_from_scores_df(df[['user_id', 'item_id', 'score']], K)

# -----------------------------
# Ground truth relevant items from ratings
# -----------------------------
def build_relevant_dict(ratings_truth_df, relevance_threshold=4.0):
    """
    ratings_truth_df: DataFrame with columns ['user_id','item_id','rating']
    returns dict: user_id -> set(item_id) considered relevant (rating >= threshold)
    """
    df = ratings_truth_df.copy()
    df = df[df['rating'] >= relevance_threshold]
    grouped = df.groupby('user_id')['item_id'].apply(set)
    return grouped.to_dict()

# -----------------------------
# End-to-end evaluation for a single model
# -----------------------------
def evaluate_model_ratings(pred_ratings_df, ratings_truth_df):
    """
    pred_ratings_df: DataFrame ['user_id','item_id','pred_rating']
    ratings_truth_df: DataFrame ['user_id','item_id','rating'] (test set)
    Returns: RMSE (float)
    """
    merged = ratings_truth_df.merge(pred_ratings_df, on=['user_id','item_id'], how='left')
    if merged['pred_rating'].isna().any():
        merged = merged.dropna(subset=['pred_rating'])
    return rmse(merged['rating'], merged['pred_rating'])

def evaluate_model_ranking(scores_df, ratings_truth_df, K_list=[5,10], relevance_threshold=4.0):
    """
    scores_df: DataFrame ['user_id','item_id','score'] produced by model (higher = more likely relevant)
    ratings_truth_df: DataFrame ['user_id','item_id','rating']
    K_list: list of K values to evaluate
    Returns: dict mapping K -> (precision, recall, f1)
    """
    relevant = build_relevant_dict(ratings_truth_df, relevance_threshold=relevance_threshold)
    results = {}
    if scores_df is None or scores_df.empty:
        # return zeros
        for K in K_list:
            results[K] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        return results

    maxK = max(K_list)
    topk_dict = topk_from_scores_df(scores_df, K=maxK)
    for K in K_list:
        recs_k = {u: topk_dict.get(u, [])[:K] for u in topk_dict}
        p, r, f = precision_recall_f1_at_k(recs_k, relevant, K)
        results[K] = {'precision': p, 'recall': r, 'f1': f}
    return results

# -----------------------------
# Visualization helpers
# -----------------------------
def plot_sentiment_distribution_by_category(df, category_col='category', sentiment_col='sentiment',
                                            figsize=(10,6), bins=40, show_mean=True):
    plt.figure(figsize=figsize)
    sns.histplot(df[sentiment_col].dropna(), bins=bins, kde=True)
    plt.title('Overall Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.show()

    categories = df[category_col].unique()
    n = len(categories)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols*5, rows*3.5))
    for i, cat in enumerate(sorted(categories)):
        plt.subplot(rows, cols, i+1)
        subset = df[df[category_col] == cat]
        if subset.empty:
            continue
        sns.histplot(subset[sentiment_col].dropna(), bins=bins, kde=True)
        if show_mean:
            mean_val = subset[sentiment_col].mean()
            plt.axvline(mean_val, color='r', linestyle='--', linewidth=1)
        plt.title(f'{cat} (n={len(subset)})')
        plt.xlabel('')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,6))
    sns.boxplot(x=category_col, y=sentiment_col, data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Sentiment distribution across categories (boxplot)')
    plt.show()

def plot_embeddings_2d(user_embs, item_embs, user_ids=None, item_ids=None, method='tsne',
                       sample_users=500, sample_items=500, random_state=42):
    # Normalize inputs to arrays and id lists
    def dict_to_arrays(d):
        ids = list(d.keys())
        mat = np.stack([d[i] for i in ids])
        return mat, ids

    if isinstance(user_embs, dict):
        U, user_ids = dict_to_arrays(user_embs)
    else:
        U = np.array(user_embs)
        if user_ids is None:
            user_ids = list(range(U.shape[0]))

    if isinstance(item_embs, dict):
        V, item_ids = dict_to_arrays(item_embs)
    else:
        V = np.array(item_embs)
        if item_ids is None:
            item_ids = list(range(V.shape[0]))

    rng = np.random.RandomState(random_state)
    u_idx = rng.choice(range(U.shape[0]), size=min(sample_users, U.shape[0]), replace=False)
    v_idx = rng.choice(range(V.shape[0]), size=min(sample_items, V.shape[0]), replace=False)

    X = np.vstack([U[u_idx], V[v_idx]])
    labels = (['user'] * len(u_idx)) + (['item'] * len(v_idx))
    ids = [user_ids[i] for i in u_idx] + [item_ids[i] for i in v_idx]

    if method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(random_state=random_state)
        X2 = reducer.fit_transform(X)
    else:
        tsne = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto')
        X2 = tsne.fit_transform(X)

    df = pd.DataFrame({
        'x': X2[:,0], 'y': X2[:,1], 'type': labels, 'id': ids
    })

    plt.figure(figsize=(10,8))
    sns.scatterplot(data=df, x='x', y='y', hue='type', alpha=0.7, s=40)
    plt.title(f'2D embedding visualization ({method.upper() if method else "TSNE"})')
    plt.legend()
    plt.show()
    return df

# -----------------------------
# New: Hybrid evaluation helpers
# -----------------------------
def load_review_embeddings(emb_path, feats_pkl_path):
    """
    Loads review embeddings (.npy) and feature-engineered reviews DataFrame (pkl).
    Returns: (df_reviews, embeddings) where embeddings[i] corresponds to df_reviews.iloc[i]
    """
    if not os.path.exists(feats_pkl_path):
        raise FileNotFoundError(f"feature-engineered pkl not found at {feats_pkl_path}")
    df = pd.read_pickle(feats_pkl_path)
    if not os.path.exists(emb_path):
        logging.warning(f"Embeddings file not found at {emb_path}. Skipping embedding-based features.")
        return df, None
    embs = np.load(emb_path)
    if len(embs) != len(df):
        logging.warning("Embeddings length != feature dataframe length. Attempting to proceed but check alignment.")
    return df, embs

def build_user_item_content_embeddings(df_reviews, review_embs):
    """
    df_reviews: DataFrame with columns ['user_id','item_id'] (and possibly others)
    review_embs: np.array shape (n_reviews, d) aligned with df_reviews rows
    Returns:
        - user_content_emb: dict user_id -> vector (mean of their review embeddings)
        - item_content_emb: dict item_id -> vector (mean of its review embeddings)
    """
    if review_embs is None:
        return {}, {}
    df = df_reviews.copy()
    # protect for missing columns
    if 'user_id' not in df.columns or 'item_id' not in df.columns:
        raise ValueError("df_reviews must contain 'user_id' and 'item_id' columns")
    # ensure same length
    n = min(len(df), len(review_embs))
    df = df.iloc[:n].reset_index(drop=True)
    embs = review_embs[:n]
    df['_emb_index'] = range(n)
    # build per-user
    user_groups = df.groupby('user_id')['_emb_index'].apply(list)
    item_groups = df.groupby('item_id')['_emb_index'].apply(list)
    user_content_emb = {}
    item_content_emb = {}
    for u, idxs in user_groups.items():
        user_content_emb[u] = np.mean(embs[idxs], axis=0)
    for it, idxs in item_groups.items():
        item_content_emb[it] = np.mean(embs[idxs], axis=0)
    return user_content_emb, item_content_emb

def compute_content_similarity_scores(user_item_pairs_df, user_content_emb, item_content_emb):
    """
    Given a DataFrame with columns ['user_id','item_id'], compute cosine similarity between user and item content embeddings.
    Returns DataFrame with column 'content_score' in [ -1, 1 ] (will be normalized later).
    """
    pairs = user_item_pairs_df.copy()
    sims = []
    for _, row in pairs.iterrows():
        uid = row['user_id']
        iid = row['item_id']
        u_emb = user_content_emb.get(uid)
        v_emb = item_content_emb.get(iid)
        if u_emb is None or v_emb is None:
            sims.append(np.nan)
        else:
            # cosine similarity (1D)
            s = float(cosine_similarity(u_emb.reshape(1,-1), v_emb.reshape(1,-1))[0,0])
            sims.append(s)
    pairs['content_score_raw'] = sims
    return pairs

def normalize_scores(df, score_col, method='global'):
    """
    Normalize a column to [0,1]. method: 'global' min-max across entire df, or 'per_user' do per user min-max.
    """
    out = df.copy()
    if method == 'global':
        vals = out[score_col].astype(float)
        # handle all NaN
        if vals.dropna().empty:
            out[score_col+'_norm'] = np.nan
            return out
        normed = minmax_scale(vals.fillna(vals.mean()))
        out[score_col+'_norm'] = normed
        return out
    elif method == 'per_user':
        out[score_col+'_norm'] = np.nan
        for uid, g in out.groupby('user_id').indices.items():
            idxs = g if isinstance(g, (list, np.ndarray)) else [g]
            vals = out.loc[idxs, score_col].astype(float)
            if vals.dropna().empty:
                out.loc[idxs, score_col+'_norm'] = np.nan
            else:
                m = vals.min()
                M = vals.max()
                if M==m:
                    out.loc[idxs, score_col+'_norm'] = 0.5
                else:
                    out.loc[idxs, score_col+'_norm'] = (vals - m) / (M - m)
        # fill remaining NaNs with global mean
        out[score_col+'_norm'] = out[score_col+'_norm'].fillna(out[score_col+'_norm'].mean())
        return out
    else:
        raise ValueError("Unknown normalization method")

def fuse_scores(df_pairs, score_columns_with_weights):
    """
    df_pairs: DataFrame with columns ['user_id','item_id', ... score columns ...]
    score_columns_with_weights: dict {score_col_name_norm: weight} where score_col_name_norm is the normalized column (e.g., 'svm_score_norm')
    Returns df_pairs with 'hybrid_score' column as weighted sum (weights will be normalized)
    """
    # filter existing columns
    present = {col: w for col,w in score_columns_with_weights.items() if col in df_pairs.columns}
    if len(present) == 0:
        raise ValueError("No score columns present to fuse.")
    # normalize weights
    total = sum(present.values())
    weights = {col: w/total for col,w in present.items()}
    hybrid = np.zeros(len(df_pairs), dtype=float)
    for col, w in weights.items():
        vals = df_pairs[col].astype(float).fillna(0.0).values
        hybrid += w * vals
    df_pairs['hybrid_score'] = hybrid
    return df_pairs

# -----------------------------
# High-level hybrid evaluation pipeline
# -----------------------------
def evaluate_hybrid_from_files(feature_pkl='../data/feature_engineered_reviews.pkl',
                               review_emb_npy='../data/review_embeddings.npy',
                               svm_csv='../data/svm_predictions.csv',
                               ncf_csv='ncf_scores.csv',
                               ratings_truth_csv=None,
                               K_list=[5,10],
                               weights=None,
                               output_dir='eval_outputs'):
    """
    Orchestrates hybrid score creation and evaluation from common file outputs.
    - weights: dict e.g. {'svm':0.3, 'ncf':0.5, 'content':0.2}; omitted keys are ignored and weights re-normalized.
    Returns: summary_df, details dict
    """
    os.makedirs(output_dir, exist_ok=True)
    models_outputs = {}
    # load ratings truth if provided
    ratings_truth_df = None
    if ratings_truth_csv and os.path.exists(ratings_truth_csv):
        ratings_truth_df = pd.read_csv(ratings_truth_csv)
    elif ratings_truth_csv:
        logging.warning("ratings_truth_csv path provided but not found. Skipping RMSE and relevance-based evaluation.")

    # 1) load feature engineered reviews + embeddings
    try:
        df_reviews, review_embs = load_review_embeddings(review_emb_npy, feature_pkl)
        logging.info("Loaded feature-engineered reviews and embeddings.")
    except Exception as e:
        logging.warning(f"Failed to load reviews/embeddings: {e}")
        df_reviews, review_embs = None, None

    # Build user-item master pairs for evaluation: union of (svm,ncf,truth) user-item pairs
    pairs = []
    # try to load svm predictions
    svm_df = None
    if svm_csv and os.path.exists(svm_csv):
        try:
            svm_df = pd.read_csv(svm_csv)
            # normalize expected columns
            if 'prob_positive' not in svm_df.columns and 'pred_label' in svm_df.columns:
                # convert label 0/1 to prob-like
                svm_df['prob_positive'] = svm_df['pred_label'].astype(float)
            if 'prob_positive' not in svm_df.columns and 'score' in svm_df.columns:
                svm_df['prob_positive'] = svm_df['score'].astype(float)
            logging.info(f"Loaded SVM predictions from {svm_csv}.")
            pairs.append(svm_df[['user_id','item_id']])
        except Exception as e:
            logging.warning(f"Failed to load SVM CSV: {e}")
            svm_df = None
    else:
        logging.info("No svm_predictions.csv found; skipping SVM component.")

    # try to load ncf scores
    ncf_df = None
    if ncf_csv and os.path.exists(ncf_csv):
        try:
            ncf_df = pd.read_csv(ncf_csv)
            if 'score' not in ncf_df.columns and 'pred_rating' in ncf_df.columns:
                ncf_df['score'] = ncf_df['pred_rating'].astype(float)
            logging.info(f"Loaded NCF scores from {ncf_csv}.")
            pairs.append(ncf_df[['user_id','item_id']])
        except Exception as e:
            logging.warning(f"Failed to load NCF CSV: {e}")
            ncf_df = None
    else:
        logging.info("No ncf_scores.csv found; skipping NCF component (if you have a .pth model, you'll need to produce score CSV).")

    # include ground truth pairs if present
    if ratings_truth_df is not None:
        pairs.append(ratings_truth_df[['user_id','item_id']])

    # build master pairs df (unique)
    if len(pairs) == 0:
        raise ValueError("No user-item pairs found from SVM/NCF/truth. Provide at least one of svm_predictions.csv, ncf_scores.csv, or ratings_truth_csv.")
    master_pairs = pd.concat(pairs, ignore_index=True).drop_duplicates().reset_index(drop=True)

    # compute content similarity if embeddings available
    user_content_emb, item_content_emb = {}, {}
    content_scores_df = None
    if df_reviews is not None and review_embs is not None:
        user_content_emb, item_content_emb = build_user_item_content_embeddings(df_reviews, review_embs)
        content_scores_df = compute_content_similarity_scores(master_pairs[['user_id','item_id']].copy(),
                                                             user_content_emb, item_content_emb)
        # normalize content similarity (-1..1) -> [0,1]
        if content_scores_df is not None:
            # fill NaN with 0
            content_scores_df['content_score_raw'] = content_scores_df['content_score_raw'].fillna(0.0)
            # map -1..1 -> 0..1
            content_scores_df['content_score'] = (content_scores_df['content_score_raw'] + 1.0) / 2.0
            master_pairs = master_pairs.merge(content_scores_df[['content_score']], left_index=True, right_index=True)
    else:
        master_pairs['content_score'] = np.nan

    # attach SVM scores to master_pairs
    if svm_df is not None:
        tmp = svm_df[['user_id','item_id','prob_positive']].copy()
        tmp = tmp.rename(columns={'prob_positive':'svm_score_raw'})
        master_pairs = master_pairs.merge(tmp, on=['user_id','item_id'], how='left')
        master_pairs['svm_score_raw'] = master_pairs['svm_score_raw'].fillna(master_pairs['svm_score_raw'].mean() if 'svm_score_raw' in master_pairs.columns else 0.0)
    else:
        master_pairs['svm_score_raw'] = np.nan

    # attach NCF scores
    if ncf_df is not None:
        tmp = ncf_df[['user_id','item_id','score']].copy().rename(columns={'score':'ncf_score_raw'})
        master_pairs = master_pairs.merge(tmp, on=['user_id','item_id'], how='left')
        master_pairs['ncf_score_raw'] = master_pairs['ncf_score_raw'].fillna(master_pairs['ncf_score_raw'].mean() if 'ncf_score_raw' in master_pairs.columns else 0.0)
    else:
        master_pairs['ncf_score_raw'] = np.nan

    # Normalize each component to [0,1]
    master_pairs = normalize_scores(master_pairs, 'svm_score_raw', method='global')
    master_pairs = normalize_scores(master_pairs, 'ncf_score_raw', method='global')
    # content_score is already in 0..1 if computed; ensure normalized column exists
    if 'content_score' in master_pairs.columns:
        master_pairs['content_score_norm'] = master_pairs['content_score'].fillna(master_pairs['content_score'].mean())
    else:
        master_pairs['content_score_norm'] = np.nan

    # Decide weights (defaults: more weight to NCF, moderate to SVM, some to content)
    default_weights = {'ncf_score_raw_norm': 0.5, 'svm_score_raw_norm': 0.3, 'content_score_norm': 0.2}
    if weights is None:
        weights = default_weights
    else:
        # map friendly keys if user passed short names
        mapped = {}
        for k,v in weights.items():
            if k.lower() in ('ncf','ncf_score','ncf_score_raw'):
                mapped['ncf_score_raw_norm'] = v
            elif k.lower() in ('svm','svm_score','svm_score_raw'):
                mapped['svm_score_raw_norm'] = v
            elif k.lower() in ('content','bert','content_score'):
                mapped['content_score_norm'] = v
            else:
                # assume they passed normalized column name already
                mapped[k] = v
        weights = mapped

    # Ensure normalized column names exist: create names used in mapping above
    # Our normalize_scores created 'svm_score_raw_norm' and 'ncf_score_raw_norm'
    # If not present, try to use existing 'svm_score_raw_norm' else map from 'svm_score_raw_norm' fallback
    # (normalize_scores already created *_norm columns)
    # Fuse
    score_cols_weights = {col: w for col,w in weights.items() if col in master_pairs.columns}
    # If user provided non-normalized keys, check for variants
    if len(score_cols_weights) == 0:
        # fallback try common names
        candidates = {}
        if 'ncf_score_raw_norm' in master_pairs.columns:
            candidates['ncf_score_raw_norm'] = weights.get('ncf_score_raw_norm', default_weights.get('ncf_score_raw_norm', 0.5))
        if 'svm_score_raw_norm' in master_pairs.columns:
            candidates['svm_score_raw_norm'] = weights.get('svm_score_raw_norm', default_weights.get('svm_score_raw_norm', 0.3))
        if 'content_score_norm' in master_pairs.columns:
            candidates['content_score_norm'] = weights.get('content_score_norm', default_weights.get('content_score_norm', 0.2))
        score_cols_weights = candidates

    # If still empty, pick whatever normalized columns exist
    if len(score_cols_weights) == 0:
        for c in ['ncf_score_raw_norm','svm_score_raw_norm','content_score_norm']:
            if c in master_pairs.columns:
                score_cols_weights[c] = 1.0

    # fuse
    master_pairs = fuse_scores(master_pairs, score_cols_weights)

    # Prepare outputs for ranking evaluation: DataFrame with columns user_id,item_id,score
    hybrid_scores_df = master_pairs[['user_id','item_id','hybrid_score']].copy().rename(columns={'hybrid_score':'score'})

    # Evaluate ranking metrics if ratings_truth_df available
    results = {}
    if ratings_truth_df is not None:
        ranking_metrics = evaluate_model_ranking(hybrid_scores_df, ratings_truth_df, K_list=K_list)
        results['ranking'] = ranking_metrics
    else:
        logging.info("ratings_truth_df not provided; skipping ranking metrics based on relevance threshold.")

    # Evaluate RMSE if NCF predicted ratings are available and ratings_truth_df provided
    if ratings_truth_df is not None and ncf_df is not None and 'pred_rating' in ncf_df.columns:
        try:
            rm = evaluate_model_ratings(ncf_df[['user_id','item_id','pred_rating']], ratings_truth_df)
            results['ncf_rmse'] = rm
        except Exception as e:
            logging.warning(f"Failed computing RMSE for NCF predictions: {e}")

    # Save outputs
    hybrid_scores_df.to_csv(os.path.join(output_dir, 'hybrid_scores.csv'), index=False)
    master_pairs.to_csv(os.path.join(output_dir, 'hybrid_components_and_scores.csv'), index=False)

    # Summarize into DataFrame
    summary_rows = []
    if 'ranking' in results:
        row = {'model':'Hybrid'}
        for K in K_list:
            row[f'P@{K}'] = results['ranking'][K]['precision']
            row[f'R@{K}'] = results['ranking'][K]['recall']
            row[f'F1@{K}'] = results['ranking'][K]['f1']
        summary_rows.append(row)
    if 'ncf_rmse' in results:
        summary_rows.append({'model':'NCF','RMSE': results['ncf_rmse']})
    summary_df = pd.DataFrame(summary_rows).set_index('model') if summary_rows else pd.DataFrame()

    # simple visualization: distribution of hybrid scores
    plt.figure(figsize=(8,5))
    sns.histplot(hybrid_scores_df['score'].dropna(), bins=50, kde=True)
    plt.title('Hybrid score distribution')
    plt.xlabel('hybrid_score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_score_distribution.png'))
    plt.show()

    # return
    details = {'master_pairs': master_pairs, 'hybrid_scores_df': hybrid_scores_df, 'user_content_emb': user_content_emb, 'item_content_emb': item_content_emb}
    return summary_df, details

# -----------------------------
# High-level evaluation pipeline (existing)
# -----------------------------
def evaluate_all_models(models_outputs, ratings_truth_df, K_list=[5,10], relevance_threshold=4.0):
    """
    models_outputs: dict, keys are model names (e.g., 'SVM', 'MiniLM', 'NCF'), values are dicts with:
        - For rating output models: {'type': 'ratings', 'pred_df': DataFrame(user_id,item_id,pred_rating)}
        - For score/ranking models: {'type': 'scores', 'scores_df': DataFrame(user_id,item_id,score)}
        - Optionally models can provide both.
    ratings_truth_df: DataFrame user_id,item_id,rating
    K_list: list of K values to compute Precision/Recall/F1
    Returns:
        summary_df: DataFrame with rows per model and columns for RMSE (if available) and ranking metrics per K.
        details: dict of per-model detailed outputs (e.g., topk dicts)
    """
    rows = []
    details = {}
    for model_name, out in models_outputs.items():
        row = {'model': model_name}
        details[model_name] = {}
        # RMSE if rating predictions provided
        if out.get('type') == 'ratings' or out.get('pred_df') is not None:
            pred_df = out.get('pred_df')
            try:
                rm = evaluate_model_ratings(pred_df, ratings_truth_df)
                row['RMSE'] = rm
            except Exception as e:
                row['RMSE'] = None
                details[model_name]['rmse_error'] = str(e)
        else:
            row['RMSE'] = None

        # Ranking metrics if score predictions provided (or if ratings provided we can use pred_rating as score)
        if out.get('type') == 'scores' or out.get('scores_df') is not None:
            scores_df = out.get('scores_df')
            ranking_metrics = evaluate_model_ranking(scores_df, ratings_truth_df, K_list, relevance_threshold)
            for K in K_list:
                row[f'P@{K}'] = ranking_metrics[K]['precision']
                row[f'R@{K}'] = ranking_metrics[K]['recall']
                row[f'F1@{K}'] = ranking_metrics[K]['f1']
            # store top-K lists
            details[model_name]['topk'] = topk_from_scores_df(scores_df, max(K_list))
        elif out.get('type') == 'ratings' and out.get('pred_df') is not None:
            # use pred_rating as score
            scores_df = out['pred_df'].rename(columns={'pred_rating':'score'})[['user_id','item_id','score']]
            ranking_metrics = evaluate_model_ranking(scores_df, ratings_truth_df, K_list, relevance_threshold)
            for K in K_list:
                row[f'P@{K}'] = ranking_metrics[K]['precision']
                row[f'R@{K}'] = ranking_metrics[K]['recall']
                row[f'F1@{K}'] = ranking_metrics[K]['f1']
            details[model_name]['topk'] = topk_from_scores_df(scores_df, max(K_list))
        else:
            for K in K_list:
                row[f'P@{K}'] = None
                row[f'R@{K}'] = None
                row[f'F1@{K}'] = None

        rows.append(row)
    summary_df = pd.DataFrame(rows).set_index('model')
    return summary_df, details
 
# -----------------------------
# Example usage template
# -----------------------------
if __name__ == "__main__":
    # Example: run hybrid evaluation from files
    # Ensure these file paths exist or adapt them
    feature_pkl = '../data/feature_engineered_reviews.pkl'
    review_emb_npy = '../data/review_embeddings.npy'
    svm_csv = 'svm_predictions.csv'           # should contain user_id,item_id,prob_positive (or pred_label)
    ncf_csv = 'ncf_scores.csv'               # should contain user_id,item_id,score (or pred_rating)
    ratings_truth_csv = 'test_ratings.csv'   # optional: for RMSE and relevance-based ranking

    # custom weights (optional); if omitted default will be used
    weights = {'ncf': 0.5, 'svm': 0.3, 'content': 0.2}

    summary, details = evaluate_hybrid_from_files(feature_pkl=feature_pkl,
                                                  review_emb_npy=review_emb_npy,
                                                  svm_csv=svm_csv,
                                                  ncf_csv=ncf_csv,
                                                  ratings_truth_csv=ratings_truth_csv,
                                                  K_list=[5,10],
                                                  weights=weights,
                                                  output_dir='eval_outputs')
    print("Hybrid evaluation summary:")
    print(summary)
