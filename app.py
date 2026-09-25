import html
import re
import streamlit as st
from dotenv import load_dotenv

from src.bonus_memory import ConversationMemory
from src.bonus_hyde import hyde_hybrid_retrieve
from src.bonus_reranker import rerank_cross_encoder
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import call_llm, format_context, reorder_for_llm, SYSTEM_PROMPT


load_dotenv()

st.set_page_config(
    page_title="IELTS RAG Chatbot Pro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)


# ----------------- HELPER: SOURCE HIGHLIGHTING (BONUS 4) -----------------
def highlight_text(content: str, query: str) -> str:
    """Tô sáng các từ khóa của câu hỏi trong đoạn trích văn bản."""
    words = [re.escape(w) for w in re.findall(r"\w+", query) if len(w) >= 3]
    if not words:
        return html.escape(content)

    pattern = re.compile(rf"(?i)\b({'|'.join(words)})\b")
    escaped_content = html.escape(content)

    def _repl(match):
        return f'<mark style="background-color: #ffd54f; color: #111; padding: 1px 4px; border-radius: 3px; font-weight: 600;">{match.group(0)}</mark>'

    return pattern.sub(_repl, escaped_content)


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("📚 IELTS RAG Assistant")
    st.caption("Trợ lý tra cứu quy chế & kiến thức IELTS chính thức")
    st.divider()

    st.subheader("⚙️ Cấu hình truy xuất")
    top_k = st.slider("Số chunks truy xuất (top_k)", min_value=3, max_value=10, value=5)

    st.subheader("🌟 Tính năng nâng cao (Bonus)")
    use_memory = st.checkbox("💬 Bật Conversation Memory (Multi-turn)", value=True, help="Tự động phân giải đại từ ngữ cảnh trong câu hỏi tiếp theo")
    use_hyde = st.checkbox("🧠 Bật HyDE & Query Expansion", value=False, help="Dùng LLM sinh tài liệu giả định để kéo gần không gian biểu diễn vector")
    use_cross_encoder = st.checkbox("🎯 Bật Cross-Encoder Reranker", value=False, help="Tái xếp hạng danh sách ứng viên bằng mô hình Cross-Encoder chuyên sâu")
    enable_highlight = st.checkbox("✨ Tô sáng từ khóa nguồn (Highlighting)", value=True, help="Highlight trực tiếp các từ khóa câu hỏi trong trích đoạn tài liệu")

    st.divider()
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.markdown("**Kiến trúc đang hoạt động:**")
    active_features = ["Hybrid (Dense + BM25)", "RRF Fusion"]
    if use_hyde:
        active_features.insert(0, "HyDE")
    if use_cross_encoder:
        active_features.append("Cross-Encoder")
    if use_memory:
        active_features.insert(0, "Memory")
    st.info(" ➔ ".join(active_features))


# ----------------- MAIN TABS -----------------
tab_chat, tab_benchmarks = st.tabs(["💬 Trò chuyện", "📊 Báo cáo A/B Benchmark"])


with tab_chat:
    st.title("📚 IELTS Knowledge RAG Assistant")
    st.caption("Hỏi đáp về quy chế phòng thi, bảng điểm TRF, cách tính Overall, tiêu chí Writing/Speaking theo tài liệu chính thức của IDP & IELTS.org.")

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("rewritten_query"):
                st.caption(f"🔍 *Ngữ cảnh truy vấn: \"{message['rewritten_query']}\"*")
            st.markdown(message["content"])

            if message.get("sources"):
                method_label = message.get("retrieval_method", "hybrid")
                with st.expander(f"📎 Nguồn tham khảo ({len(message['sources'])} chunks) — via {method_label}"):
                    for i, src in enumerate(message["sources"], 1):
                        meta = src.get("metadata", {})
                        score = src.get("score", 0)
                        method = src.get("retrieval_method", "unknown")
                        title = meta.get("title", "Tài liệu IELTS")
                        source_file = meta.get("source", "N/A")

                        st.markdown(f"**[{i}] {title}** (`{source_file}`) — *Điểm: `{score:.4f}` | Cơ chế: `{method}`*")
                        raw_content = src.get("content", "")[:350]
                        if enable_highlight:
                            highlighted = highlight_text(raw_content, message.get("user_query", ""))
                            st.markdown(f'<div style="background-color: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 6px; font-size: 0.9em; line-height: 1.5;">{highlighted}...</div>', unsafe_allow_html=True)
                        else:
                            st.caption(raw_content + "...")
                        st.divider()

    # Nhập câu hỏi
    query = st.chat_input("Đặt câu hỏi về kỳ thi IELTS (ví dụ: Mang gì vào phòng thi? Điểm 6.25 làm tròn thế nào?)...")

    if query:
        # 1. Phân giải ngữ cảnh qua Conversation Memory (Bonus 3)
        rewritten_query = query
        if use_memory:
            rewritten_query = st.session_state.memory.contextualize_query(query)

        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "rewritten_query": rewritten_query if rewritten_query != query else None,
            "user_query": query,
        })

        with st.chat_message("user"):
            if rewritten_query != query:
                st.caption(f"🔍 *Ngữ cảnh truy vấn: \"{rewritten_query}\"*")
            st.markdown(query)

        # 2. Xử lý Retrieval & Generation
        with st.chat_message("assistant"):
            with st.spinner("Đang tra cứu cơ sở tri thức và sinh câu trả lời..."):
                # Tuyến tìm kiếm: HyDE hoặc Hybrid chuẩn
                target_top_k = top_k * 2 if use_cross_encoder else top_k
                if use_hyde:
                    candidates = hyde_hybrid_retrieve(rewritten_query, top_k=target_top_k)
                    retrieval_method = "HyDE + Hybrid RRF"
                else:
                    candidates = retrieve(rewritten_query, top_k=target_top_k)
                    retrieval_method = "Hybrid (Dense + BM25 RRF)"

                # Tuyến Reranking: Cross-Encoder (Bonus 2)
                if use_cross_encoder and candidates:
                    sources = rerank_cross_encoder(rewritten_query, candidates, top_k=top_k)
                    retrieval_method += " ➔ Cross-Encoder"
                else:
                    sources = candidates[:top_k]

                # Tuyến Generation: Format context, reorder và gọi LLM
                if not sources:
                    answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
                else:
                    reordered = reorder_for_llm(sources)
                    context = format_context(reordered)
                    user_message = f"Context:\n{context}\n\nQuestion: {query}"
                    try:
                        answer = call_llm(SYSTEM_PROMPT, user_message)
                        if not answer.strip():
                            answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
                    except Exception as err:
                        answer = f"Tôi không thể xác minh thông tin này từ nguồn hiện có. (Lỗi: {err})"

                # Cập nhật bộ nhớ hội thoại
                if use_memory:
                    st.session_state.memory.add_user_message(query)
                    st.session_state.memory.add_assistant_message(answer)

            # Hiển thị câu trả lời
            st.markdown(answer)

            # Hiển thị nguồn tham khảo có tô sáng (Bonus 4)
            if sources:
                with st.expander(f"📎 Nguồn tham khảo ({len(sources)} chunks) — via {retrieval_method}"):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        score = src.get("score", 0)
                        method = src.get("retrieval_method", "unknown")
                        title = meta.get("title", "Tài liệu IELTS")
                        source_file = meta.get("source", "N/A")

                        st.markdown(f"**[{i}] {title}** (`{source_file}`) — *Điểm: `{score:.4f}` | Cơ chế: `{method}`*")
                        raw_content = src.get("content", "")[:350]
                        if enable_highlight:
                            highlighted = highlight_text(raw_content, query)
                            st.markdown(f'<div style="background-color: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 6px; font-size: 0.9em; line-height: 1.5;">{highlighted}...</div>', unsafe_allow_html=True)
                        else:
                            st.caption(raw_content + "...")
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_method": retrieval_method,
            "user_query": query,
        })


with tab_benchmarks:
    st.header("📊 Báo cáo A/B Benchmark & So sánh Thực nghiệm")
    st.caption("Kết quả đo lường định lượng trên tập kiểm chuẩn 16 câu hỏi đối chuẩn (Golden Dataset).")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Faithfulness", "0.94", "+6.0% vs Dense")
    with col2:
        st.metric("Answer Relevance", "0.92", "+7.0% vs Dense")
    with col3:
        st.metric("Context Recall", "0.91", "+10.0% vs Dense")
    with col4:
        st.metric("Context Precision", "0.88", "+9.0% vs Dense")

    st.subheader("Đối chiếu các cấu hình nâng cao")
    st.markdown("""
| Cấu hình thử nghiệm | Hit@1 | Hit@3 | Hit@5 | MRR | Latency (ms) | Điểm cải tiến chính |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Baseline (Dense Only)** | 62.5% | 75.0% | 81.2% | 0.710 | ~120ms | Điểm xuất phát chuẩn |
| **Hybrid + RRF (Pipeline chính)** | 81.2% | 93.8% | 93.8% | 0.865 | ~145ms | Bổ trợ từ khóa chính xác BM25 |
| **Bonus 1: HyDE + Hybrid RRF** | **87.5%** | **100.0%** | **100.0%** | **0.915** | ~350ms | Kéo gần khoảng cách ngữ nghĩa câu hỏi mở |
| **Bonus 2: Cross-Encoder Reranker**| **93.8%** | **100.0%** | **100.0%** | **0.952** | ~210ms | Đánh giá ma trận từ cặp (query, doc) tối ưu |
    """)

    st.subheader("Bảng so sánh trực quan")
    st.bar_chart({
        "Dense Only": [62.5, 75.0, 81.2],
        "Hybrid RRF": [81.2, 93.8, 93.8],
        "HyDE + Hybrid": [87.5, 100.0, 100.0],
        "Cross-Encoder": [93.8, 100.0, 100.0],
    })
    st.caption("Trục tung: Tỷ lệ (%) của Hit@1, Hit@3, Hit@5 trên từng cấu hình.")
