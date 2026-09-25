"""
A/B Benchmark Script — Đánh giá định lượng tính năng Bonus:
1. Baseline: Hybrid Retrieval (Dense + BM25) + RRF
2. Challenger 1: HyDE (Hypothetical Document Embeddings) + Hybrid + RRF
3. Challenger 2: Hybrid Retrieval + Cross-Encoder Reranker

Đo lường trên 16 câu hỏi đối chuẩn của golden_dataset.json:
- Hit@1: Tỷ lệ context chuẩn nằm ở top 1
- Hit@3: Tỷ lệ context chuẩn nằm trong top 3
- Hit@5: Tỷ lệ context chuẩn nằm trong top 5
- MRR (Mean Reciprocal Rank): Điểm trung bình nghịch đảo thứ hạng
- Latency (ms): Thời gian truy xuất trung bình
"""

import json
import time
from pathlib import Path

from src.task9_retrieval_pipeline import retrieve
from src.bonus_hyde import hyde_hybrid_retrieve
from src.bonus_reranker import rerank_cross_encoder
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search


GOLDEN_PATH = Path(__file__).parent.parent / "group_project" / "evaluation" / "golden_dataset.json"


def evaluate_ranking(results: list[dict], expected_context: str, expected_answer: str) -> tuple[float, float, float, float]:
    """Tính Hit@1, Hit@3, Hit@5 và Reciprocal Rank dựa trên độ trùng khớp nội dung ngữ nghĩa."""
    stop_words = {
        "trong", "những", "được", "người", "không", "thường", "theo", "khoảng", "các", "cho",
        "của", "và", "phải", "trên", "dưới", "ielts", "thi", "vào", "bài", "sinh", "thí",
        "một", "có", "là", "khi", "với", "đến", "này", "đó", "như", "đã", "sẽ", "lại", "ra",
        "thì", "nếu", "hợp", "cấp", "tất", "đều", "hoặc", "bao", "gồm", "khác", "nào", "gì"
    }
    raw_tokens = [w.strip(".,;:?!\"'()[]{}«»/").lower() for w in (expected_context + " " + expected_answer).split()]
    target_words = {w for w in raw_tokens if len(w) >= 3 and w not in stop_words}

    first_rank = None
    for rank, item in enumerate(results, 1):
        content_lower = item["content"].lower()
        # Kiểm tra xem có chứa đoạn văn bản ground truth hoặc tỷ lệ từ khóa đặc trưng đủ cao
        matched = sum(1 for w in target_words if w in content_lower)
        threshold = max(3, int(len(target_words) * 0.25))
        if matched >= threshold or expected_context[:35].lower() in content_lower:
            first_rank = rank
            break

    hit1 = 1.0 if first_rank == 1 else 0.0
    hit3 = 1.0 if first_rank is not None and first_rank <= 3 else 0.0
    hit5 = 1.0 if first_rank is not None and first_rank <= 5 else 0.0
    rr = 1.0 / first_rank if first_rank is not None else 0.0
    return hit1, hit3, hit5, rr


def run_benchmark():
    dataset = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"Bắt đầu A/B Benchmark trên {len(dataset)} câu hỏi đối chuẩn...\n")

    configs = {
        "Baseline (Hybrid + RRF)": lambda q: retrieve(q, top_k=5),
        "Bonus 1 (HyDE + Hybrid + RRF)": lambda q: hyde_hybrid_retrieve(q, top_k=5),
        "Bonus 2 (Cross-Encoder Reranker)": lambda q: rerank_cross_encoder(
            q,
            semantic_search(q, top_k=10) + lexical_search(q, top_k=10),
            top_k=5,
        ),
    }

    metrics = {
        name: {"hit1": 0.0, "hit3": 0.0, "hit5": 0.0, "mrr": 0.0, "latency": 0.0}
        for name in configs
    }

    n_samples = len(dataset)

    for i, case in enumerate(dataset, 1):
        query = case["question"]
        expected_context = case["expected_context"]
        expected_answer = case["expected_answer"]
        print(f"[{i:02d}/{n_samples:02d}] Đang kiểm tra: {query[:45]}...")

        for name, func in configs.items():
            start_t = time.perf_counter()
            try:
                results = func(query)
            except Exception as e:
                results = []
            dur_ms = (time.perf_counter() - start_t) * 1000.0

            h1, h3, h5, rr = evaluate_ranking(results, expected_context, expected_answer)
            metrics[name]["hit1"] += h1
            metrics[name]["hit3"] += h3
            metrics[name]["hit5"] += h5
            metrics[name]["mrr"] += rr
            metrics[name]["latency"] += dur_ms

    print("=========================================================================")
    print("                      KẾT QUẢ A/B BENCHMARK RAG                         ")
    print("=========================================================================")
    print(f"{'Cấu hình':<35} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'Latency':<8}")
    print("-" * 85)

    summary_rows = []
    for name, m in metrics.items():
        h1 = (m["hit1"] / n_samples) * 100.0
        h3 = (m["hit3"] / n_samples) * 100.0
        h5 = (m["hit5"] / n_samples) * 100.0
        mrr = m["mrr"] / n_samples
        lat = m["latency"] / n_samples
        print(f"{name:<35} | {h1:>6.1f}% | {h3:>6.1f}% | {h5:>6.1f}% | {mrr:>8.4f} | {lat:>6.1f}ms")
        summary_rows.append({
            "name": name,
            "hit1": h1,
            "hit3": h3,
            "hit5": h5,
            "mrr": mrr,
            "latency": lat,
        })
    print("=========================================================================\n")
    return summary_rows


if __name__ == "__main__":
    run_benchmark()
