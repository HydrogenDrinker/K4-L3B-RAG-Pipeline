"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_cache.json"


def _load_cache() -> dict:
    """Load cached document IDs."""
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    """Save document IDs to cache."""
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY not set, skipping upload.")
        return

    try:
        from pageindex import PageIndex

        client = PageIndex(api_key=PAGEINDEX_API_KEY)
        cache = _load_cache()

        for path in STANDARDIZED_DIR.rglob("*.md"):
            source_key = path.relative_to(STANDARDIZED_DIR).as_posix()
            if source_key in cache:
                print(f"Already uploaded: {source_key}")
                continue

            content = path.read_text(encoding="utf-8")
            if not content.strip():
                continue

            try:
                result = client.upload(content=content, filename=path.name)
                doc_id = getattr(result, "id", None) or str(result)
                cache[source_key] = doc_id
                print(f"Uploaded: {source_key} -> {doc_id}")
            except Exception as error:
                print(f"Upload failed for {source_key}: {error}")

        _save_cache(cache)
    except ImportError:
        print("pageindex package not available, skipping upload.")
    except Exception as error:
        print(f"PageIndex upload error: {error}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if not PAGEINDEX_API_KEY:
        return []

    try:
        from pageindex import PageIndex

        client = PageIndex(api_key=PAGEINDEX_API_KEY)
        cache = _load_cache()

        if not cache:
            print("No documents uploaded to PageIndex.")
            return []

        doc_ids = list(cache.values())
        response = client.query(query=query, document_ids=doc_ids, top_k=top_k)

        results = []
        nodes = getattr(response, "nodes", []) or []

        for rank, node in enumerate(nodes[:top_k]):
            content = getattr(node, "text", "") or getattr(node, "content", "") or ""
            score = getattr(node, "score", None)
            if score is None:
                score = 1.0 / (rank + 1)

            source_name = ""
            for source_key, doc_id in cache.items():
                if doc_id == getattr(node, "document_id", None):
                    source_name = source_key
                    break

            results.append({
                "id": f"pageindex-{rank}",
                "content": content,
                "score": float(score),
                "metadata": {
                    "source": source_name or "pageindex",
                    "title": source_name.replace("/", " - ") if source_name else "PageIndex",
                    "doc_type": "legal" if "legal" in source_name else "news",
                    "url": None,
                    "chunk_index": rank,
                },
                "retrieval_method": "pageindex",
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    except Exception as error:
        print(f"PageIndex search error: {error}")
        return []


if __name__ == "__main__":
    upload_documents()
