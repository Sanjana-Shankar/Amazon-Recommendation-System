import os
from typing import List, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

_BERT_TOKENIZER = None
_BERT_MODEL = None

def get_device(device: Optional[str] = None) -> str:
    if device:
        print(f"[BERT] Using explicitly provided device: {device}")
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"

def init_bert_encoder(model_name: str = "bert-base-uncased", device: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModel]:
    """
    Lazy-init and cache HF tokenizer + model. Model moved to device and set to eval().
    """
    global _BERT_TOKENIZER, _BERT_MODEL
    if _BERT_TOKENIZER is not None and _BERT_MODEL is not None:
        return _BERT_TOKENIZER, _BERT_MODEL

    print("[BERT] Loading tokenizer:", model_name)
    device = get_device(device)
    _BERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _BERT_MODEL = AutoModel.from_pretrained(model_name).to(device)
    _BERT_MODEL.eval()
    return _BERT_TOKENIZER, _BERT_MODEL

def encode_texts_bert(
    texts: List[str],
    model_name: str = "bert-base-uncased",
    batch_size: int = 32,
    pooling: str = "mean",
    normalize: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Encode a list of texts with a HuggingFace BERT-like model.
    pooling: "mean" (mean pooling over tokens with attention mask) or "cls" (CLS token).
    Returns float32 numpy array shape (len(texts), hidden_size).
    """
    tokenizer, model = init_bert_encoder(model_name=model_name, device=device)
    device = next(model.parameters()).device
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = out.last_hidden_state  # (B, L, D)

            if pooling == "cls":
                vec = last_hidden[:, 0, :]  # CLS token
            else:
                mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
                summed = (last_hidden * mask).sum(1)
                counts = mask.sum(1).clamp(min=1e-9)
                vec = summed / counts

            if normalize:
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)

            all_embs.append(vec.cpu().numpy().astype("float32"))

    if not all_embs:
        return np.zeros((0, model.config.hidden_size), dtype="float32")
    return np.vstack(all_embs)

def encode_in_chunks_and_save_bert(
    texts_iter: Iterable[str],
    out_path: str,
    model_name: str = "bert-base-uncased",
    chunk_size: int = 2000,
    batch_size: int = 32,
    pooling: str = "mean",
    normalize: bool = True,
    overwrite: bool = True,
    device: Optional[str] = None,
) -> str:
    """
    Encode texts in chunks and save to out_path as a .npy memmap. Returns out_path.
    Note: converts texts_iter -> list to know total length. For huge datasets implement streaming.
    """
    print("Encoding texts with BERT...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    texts = list(texts_iter)
    n = len(texts)
    if n == 0:
        raise ValueError("No texts to encode")

    if os.path.exists(out_path) and not overwrite:
        print("[BERT] File exists → reusing:", out_path)
        return out_path

    _, model = init_bert_encoder(model_name=model_name, device=device)
    emb_dim = model.config.hidden_size

    mmap = np.memmap(out_path, dtype="float32", mode="w+", shape=(n, emb_dim))

    for start in tqdm(range(0, n, chunk_size), desc="BERT encoding chunks"):
        end = min(start + chunk_size, n)
        chunk_texts = texts[start:end]
        chunk_emb = encode_texts_bert(
            chunk_texts,
            model_name=model_name,
            batch_size=batch_size,
            pooling=pooling,
            normalize=normalize,
            device=device,
        )
        mmap[start:end, :] = chunk_emb
        mmap.flush()

    del mmap
    return out_path

def load_embeddings(path: str, mmap_mode: Optional[str] = None) -> np.ndarray:
    """
    Load saved embeddings (.npy). If mmap_mode provided (e.g., 'r') returns a memmap.
    """
    if mmap_mode:
        return np.memmap(path, dtype="float32", mode=mmap_mode)
    return np.load(path, mmap_mode=mmap_mode)

def add_embedding_file_and_index(df: pd.DataFrame, embedding_path: str, index_col: str = "embedding_index", file_col: str = "bert_embedding_file"):
    """
    Return a copy of df with embedding file path and per-row index attached.
    """
    out = df.copy()
    out[file_col] = embedding_path
    out[index_col] = np.arange(len(out), dtype=np.int32)
    return out

def run_bert_embeddings(
    df: pd.DataFrame,
    text_col: str = "reviewText",
    out_path: str = "../data/review_embeddings.npy",
    model_name: str = "bert-base-uncased",
    chunk_size: int = 2000,
    batch_size: int = 32,
    pooling: str = "mean",
    normalize: bool = True,
    overwrite: bool = True,
    device: Optional[str] = None,
    save_mapping: bool = True,
) -> pd.DataFrame:
    """
    Encode texts from `df[text_col]` using the BERT encoder and save embeddings to out_path (memmap .npy).
    Returns a copy of df with 'bert_embedding_file' and 'embedding_index' columns attached.

    Parameters are passed down to encode_in_chunks_and_save_bert.
    """
    texts = df[text_col].fillna("").tolist()
    out = encode_in_chunks_and_save_bert(
        texts_iter=texts,
        out_path=out_path,
        model_name=model_name,
        chunk_size=chunk_size,
        batch_size=batch_size,
        pooling=pooling,
        normalize=normalize,
        overwrite=overwrite,
        device=device,
    )
    # Ensure correct file format for embeddings
    if out_path.endswith(".npy"):
        raise ValueError("Use .memmap or .bin for BERT embeddings, not .npy")

    df_map = add_embedding_file_and_index(df, out_path)
    if save_mapping:
        map_path = os.path.splitext(out_path)[0] + "_mapping.pkl"
        df_map[["bert_embedding_file", "embedding_index"]].to_pickle(map_path)
    return df_map

