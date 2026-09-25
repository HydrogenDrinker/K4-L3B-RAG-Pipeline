"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

import json
from pathlib import Path

from markitdown import MarkItDown


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    """Convert PDF/DOCX vào standardized/legal."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = MarkItDown()
    converted = 0

    for path in legal_dir.iterdir():
        if path.suffix.lower() in {".pdf", ".doc", ".docx"}:
            output_path = output_dir / f"{path.stem}.md"
            result = converter.convert(str(path))
            content = result.text_content or ""
            if not content.strip():
                print(f"Warning: empty conversion for {path.name}")
                continue
            output_path.write_text(content, encoding="utf-8")
            converted += 1
            print(f"Converted: {path.name} -> {output_path.name}")

    print(f"Legal docs converted: {converted}")


def convert_news_articles() -> None:
    """Convert JSON vào standardized/news."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0

    for path in news_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))

        content_md = data.get("content_markdown", "")
        if not content_md.strip():
            print(f"Warning: empty content in {path.name}")
            continue

        header = (
            f"# {data.get('title', 'Untitled')}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
        )

        output_path = output_dir / f"{path.stem}.md"
        output_path.write_text(header + content_md, encoding="utf-8")
        converted += 1
        print(f"Converted: {path.name} -> {output_path.name}")

    print(f"News articles converted: {converted}")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
