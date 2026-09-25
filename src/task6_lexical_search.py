"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import math
import numpy as np
from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import get_collection


class BM25(BM25Okapi):
    """BM25 với Lucene non-negative IDF để hoạt động chính xác với mọi kích thước corpus."""

    def _calc_idf(self, nd):
        self.idf = {
            word: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for word, freq in nd.items()
        }


# Module-level corpus and cache
CORPUS: list[dict] = []
_bm25_index = None
_cached_corpus_id = None


def _load_corpus() -> list[dict]:
    """Load corpus chunks từ ChromaDB."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    data = collection.get(include=["documents", "metadatas"])
    corpus = []
    for item_id, content, metadata in zip(
        data["ids"], data["documents"], data["metadatas"]
    ):
        if metadata.get("url") == "":
            metadata["url"] = None
        corpus.append({
            "id": item_id,
            "content": content,
            "metadata": metadata,
        })
    return corpus


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    tokenized = [item["content"].lower().split() for item in corpus]
    return BM25(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    global _bm25_index, _cached_corpus_id, CORPUS

    corpus = CORPUS if CORPUS else _load_corpus()
    if not corpus:
        return []

    if _bm25_index is None or _cached_corpus_id != id(corpus):
        _bm25_index = build_bm25_index(corpus)
        _cached_corpus_id = id(corpus)

    scores = _bm25_index.get_scores(query.lower().split())
    indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in indices:
        if scores[index] <= 0:
            continue
        item = corpus[index]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
    return results


if __name__ == "__main__":
    for result in lexical_search("IELTS test", top_k=3):
        print(f"[{result['score']:.4f}] {result['id']}: {result['content'][:80]}...")
