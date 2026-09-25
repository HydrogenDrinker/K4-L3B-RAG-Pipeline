"""
Bonus 1 — HyDE (Hypothetical Document Embeddings) & Query Expansion.

Kỹ thuật:
    1. HyDE: Dùng LLM sinh một đoạn văn bản giả định (hypothetical document) trả lời câu hỏi,
       sau đó embed đoạn giả định này để tìm kiếm trong ChromaDB. Giúp đưa query về cùng
       không gian biểu diễn với tài liệu corpus.
    2. Query Expansion: Sinh các câu hỏi tương đương / mở rộng từ viết tắt (TRF, EOR...)
       để tăng độ phủ từ khóa cho BM25.
"""

from .task4_chunking_indexing import embed_texts, get_collection
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task10_generation import call_llm


HYDE_PROMPT = """Bạn là chuyên gia về kỳ thi IELTS. Hãy viết một đoạn văn bản ngắn (2-3 câu) trả lời câu hỏi sau một cách thực tế và khách quan như trong tài liệu chính thức. Không giải thích thêm.

Câu hỏi: {query}
Đoạn văn bản giả định:"""

EXPANSION_PROMPT = """Hãy đưa ra 2 cách diễn đạt khác hoặc từ khóa tương đương cho câu hỏi về kỳ thi IELTS sau đây (mở rộng từ viết tắt nếu có). Mỗi câu một dòng, không đánh số:

Câu hỏi: {query}"""


def generate_hypothetical_document(query: str) -> str:
    """Sinh đoạn văn bản giả định bằng LLM."""
    try:
        hypo = call_llm("Bạn là chuyên gia IELTS.", HYDE_PROMPT.format(query=query))
        return hypo.strip() if hypo.strip() else query
    except Exception:
        return query


def expand_query(query: str) -> list[str]:
    """Mở rộng câu hỏi thành các biến thể từ khóa."""
    queries = [query]
    try:
        response = call_llm("Bạn là chuyên gia ngôn ngữ.", EXPANSION_PROMPT.format(query=query))
        lines = [line.strip().lstrip("1234567890.- ") for line in response.split("\n") if line.strip()]
        queries.extend(lines[:2])
    except Exception:
        pass
    return list(dict.fromkeys(queries))


def hyde_search(query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm bằng vector của Hypothetical Document."""
    hypo_text = generate_hypothetical_document(query)
    query_vector = embed_texts([hypo_text])[0]
    collection = get_collection()

    response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        if metadata.get("url") == "":
            metadata["url"] = None
        results.append({
            "id": item_id,
            "content": content,
            "score": max(0.0, 1.0 - distance),
            "metadata": metadata,
            "retrieval_method": "dense",
        })

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def hyde_hybrid_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Kết hợp HyDE Dense Search + Query Expansion BM25 qua RRF."""
    # 1. HyDE search
    dense_results = hyde_search(query, top_k=top_k * 2)

    # 2. Expanded BM25 search
    expanded_queries = expand_query(query)
    sparse_lists = [lexical_search(q, top_k=top_k * 2) for q in expanded_queries]
    
    # 3. Fuse HyDE + all BM25 variants
    all_ranked_lists = [dense_results] + [s for s in sparse_lists if s]
    fused = rerank_rrf(all_ranked_lists, top_k=top_k)
    return fused


if __name__ == "__main__":
    test_q = "Quy định mang nước vào phòng thi IELTS"
    print("Hypo doc:", generate_hypothetical_document(test_q))
    results = hyde_hybrid_retrieve(test_q, top_k=3)
    for r in results:
        print(f"[{r['score']:.4f}] {r['id']}")
