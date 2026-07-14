"""
Retrieval embeddings for TimeCAP's in-context example selection.

The original TimeCAP encoder (encoder/model.py, encoder/exp.py) trains a
supervised time-series + text classifier per dataset and reuses its learned
representation for retrieval. Doing that per zipcode here would mean
training a separate PyTorch model per region, which doesn't scale to
region-level forecasting with many regions. Instead we embed each region's
gpt_summary text directly with a pretrained sentence-transformer (no
training required) and use cosine similarity over that embedding space for
retrieval - the same embedding backend the original Model class already
supports as one of its `lm_model` options, just used directly instead of
behind a trained classification head.
"""

import hashlib

import numpy as np

_MODEL_CACHE = {}


def _get_model(name):
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def _hash_embed(texts, dim=64):
    """Deterministic, dependency-free embedding used only in --dry-run mode."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
            out[i, h % dim] += 1.0
    return out


def embed_texts(texts, cfg):
    texts = list(texts)
    if cfg.dry_run:
        return _hash_embed(texts)
    model = _get_model(cfg.embedding_model)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def top_k_indices(query_vec, pool_vecs, k):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    p = pool_vecs / (np.linalg.norm(pool_vecs, axis=1, keepdims=True) + 1e-8)
    sims = p @ q
    order = np.argsort(-sims)[:k]
    return order, sims[order]
