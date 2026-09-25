"""
Bonus 3 — Multi-turn Conversation Memory & Query Reformulation.

Hỗ trợ follow-up questions trong hội thoại đa lượt (Multi-turn RAG).
Kỹ thuật:
    1. Lưu trữ buffer lịch sử hỏi đáp (user, assistant).
    2. Coreference Resolution & Query Rewriting: Tự động phân giải đại từ thay thế
       (ví dụ: "Còn thi trên giấy thì sao?", "Bao nhiêu ngày có kết quả?") thành
       câu hỏi độc lập đầy đủ thực thể để retrieval pipeline tìm kiếm chính xác.
"""

from .task10_generation import call_llm


REWRITE_PROMPT = """Bạn là trợ lý xử lý ngôn ngữ tự nhiên. Dưới đây là lịch sử cuộc trò chuyện giữa người dùng và trợ lý về kỳ thi IELTS:

{history}

Người dùng vừa hỏi câu tiếp theo: "{query}"

Nhiệm vụ: Hãy viết lại câu hỏi trên thành một câu hỏi ĐỘC LẬP, RÕ RÀNG, ĐẦY ĐỦ CHỦ NGỮ và NGỮ CẢNH để hệ thống tra cứu tài liệu có thể tìm kiếm chính xác mà không cần xem lại lịch sử.
Nếu câu hỏi đã đầy đủ thông tin hoặc là một chủ đề mới hoàn toàn, hãy giữ nguyên.
CHỈ TRẢ VỀ CÂU HỎI ĐÃ VIẾT LẠI, KHÔNG GIẢI THÍCH THÊM:"""


class ConversationMemory:
    """Quản lý bộ nhớ hội thoại và viết lại truy vấn phụ thuộc ngữ cảnh."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add_user_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_assistant_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def _trim(self) -> None:
        # Giữ tối đa 2 * max_turns tin nhắn (mỗi turn gồm user + assistant)
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_history_text(self) -> str:
        """Chuyển lịch sử thành dạng văn bản ngắn gọn."""
        if not self.history:
            return ""
        lines = []
        for msg in self.history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            # Rút gọn câu trả lời dài của assistant để tiết kiệm token
            content = msg["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def contextualize_query(self, query: str) -> str:
        """Phân giải ngữ cảnh và viết lại query cho retrieval."""
        history_text = self.get_history_text()
        if not history_text.strip():
            return query

        try:
            prompt = REWRITE_PROMPT.format(history=history_text, query=query)
            rewritten = call_llm("Bạn là chuyên gia xử lý ngôn ngữ tự nhiên.", prompt)
            rewritten = rewritten.strip().strip('"').strip("'")
            if rewritten and len(rewritten) > 3:
                return rewritten
            return query
        except Exception:
            return query

    def clear(self) -> None:
        """Xóa toàn bộ lịch sử hội thoại."""
        self.history = []


if __name__ == "__main__":
    mem = ConversationMemory()
    mem.add_user_message("Thời gian nhận kết quả thi IELTS trên máy tính là bao lâu?")
    mem.add_assistant_message("Kết quả thi IELTS trên máy tính thường có trong 3 đến 5 ngày sau ngày thi.")
    
    follow_up = "Còn thi trên giấy thì sao?"
    rewritten = mem.contextualize_query(follow_up)
    print("Original:", follow_up)
    print("Rewritten:", rewritten)
