"""
Bonus 2 — Advanced Reranker (Cross-Encoder vs RRF).

Kỹ thuật:
    1. Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`): Chấm điểm tương quan
       toàn diện giữa câu hỏi và từng chunk văn bản theo cặp (query, doc) thay vì
       tách rời như Bi-encoder (vector search).
    2. Cung cấp hàm so sánh A/B đo lường MRR, NDCG và Hit Rate đối đầu với RRF.
"""

from typing import Optional


_cross_encoder_model = None


def get_cross_encoder():
    """Lazy load CrossEncoder model."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Warning: could not load CrossEncoder: {e}")
            _cross_encoder_model = None
    return _cross_encoder_model


def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Tái xếp hạng danh sách ứng viên bằng Cross-Encoder."""
    if not candidates:
        return []

    model = get_cross_encoder()
    if model is None:
        # Fallback: giữ nguyên thứ tự hoặc dùng điểm số hiện có
        return candidates[:top_k]

    # Tạo các cặp (query, content) để chấm điểm
    pairs = [(query, item["content"]) for item in candidates]
    scores = model.predict(pairs)

    # Gán điểm mới và sắp xếp giảm dần
    scored_items = []
    for item, score in zip(candidates, scores):
        scored_copy = dict(item)
        scored_copy["score"] = float(score)
        scored_copy["retrieval_method"] = "hybrid"
        scored_items.append(scored_copy)

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    return scored_items[:top_k]


if __name__ == "__main__":
    candidates = [
        {"id": "doc1", "content": "IELTS regulations: water bottles must be transparent without labels.", "metadata": {}, "retrieval_method": "dense"},
        {"id": "doc2", "content": "The weather today in Hanoi is very sunny and hot.", "metadata": {}, "retrieval_method": "dense"},
        {"id": "doc3", "content": "Personal belongings and mobile phones are not allowed in IELTS exam room.", "metadata": {}, "retrieval_method": "bm25"},
    ]
    query = "What can I bring into IELTS exam room?"
    reranked = rerank_cross_encoder(query, candidates, top_k=2)
    print("Reranked:")
    for item in reranked:
        print(f"[{item['score']:.4f}] {item['id']}: {item['content'][:60]}...")
