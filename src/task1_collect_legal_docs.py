"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

import asyncio
from pathlib import Path

from crawl4ai import AsyncWebCrawler


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# 3 trang chính sách IELTS — crawl rồi tạo PDF bằng fpdf2
LEGAL_URLS = {
    "ielts-regulations": "https://ielts.idp.com/vietnam/about/news-and-articles/article-ielts-regulations",
    "ielts-lexical-resource": "https://ielts.idp.com/vietnam/about/news-and-articles/article-l-is-for-lexical-resource",
    "ielts-coherence-cohesion": "https://ielts.idp.com/vietnam/about/news-and-articles/article-coherence-and-cohesion",
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def _save_as_pdf(filename: str, title: str, content: str) -> None:
    """Lưu nội dung text thành file PDF bằng fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Dùng font hỗ trợ Unicode
    font_path = Path(__file__).parent.parent / "fonts" / "DejaVuSans.ttf"
    if font_path.exists():
        pdf.add_font("DejaVu", "", str(font_path))
        pdf.set_font("DejaVu", size=10)
    elif Path("C:/Windows/Fonts/arial.ttf").exists():
        pdf.add_font("Arial", "", "C:/Windows/Fonts/arial.ttf")
        pdf.set_font("Arial", size=10)
    else:
        pdf.set_font("Helvetica", size=10)

    # Title
    pdf.set_font_size(16)
    pdf.multi_cell(pdf.epw, 10, title)
    pdf.ln(5)

    # Body
    pdf.set_font_size(10)
    for line in content.split("\n"):
        pdf.multi_cell(pdf.epw, 5, line)

    output_path = DATA_DIR / f"{filename}.pdf"
    pdf.output(str(output_path))
    print(f"Saved PDF: {output_path}")


async def _crawl_and_save() -> None:
    """Crawl từng URL và lưu thành PDF."""
    async with AsyncWebCrawler() as crawler:
        for filename, url in LEGAL_URLS.items():
            try:
                result = await crawler.arun(url=url)
                title = result.metadata.get("title", filename) if result.metadata else filename
                content = result.markdown or ""
                if not content.strip():
                    print(f"Warning: empty content from {url}")
                    continue
                _save_as_pdf(filename, title, content)
            except Exception as error:
                print(f"Failed: {url} — {error}")


def download_documents() -> None:
    """Crawl 3 trang chính sách IELTS và lưu thành PDF."""
    asyncio.run(_crawl_and_save())


if __name__ == "__main__":
    setup_directory()
    download_documents()
