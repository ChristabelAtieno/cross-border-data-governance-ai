from functools import lru_cache
from typing import List, Dict, Any


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)

def rerank_hits(query: str, hits: List[Dict[str, Any]], top_k: int = 10):
    if not hits:
        return []
    cross_encoder = get_cross_encoder()
    pairs = [(query, hit["content"]) for hit in hits]
    scores = cross_encoder.predict(pairs)
    
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    
    hits_sorted = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    return hits_sorted[:top_k or len(hits_sorted)]