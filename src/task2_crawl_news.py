"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from crawl4ai import AsyncWebCrawler


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://www.idp.com/vietnam/ielts/test-format-and-benefits/",
    "https://ielts.idp.com/vietnam/about/test-day",
    "https://ielts.idp.com/vietnam/results/scores",
    "https://ielts.idp.com/vietnam/about/news-and-articles/article-grammatical-range-accuracy",
    "https://ielts.org/take-a-test/preparation-resources/on-test-day",
]


async def crawl_article(url: str) -> dict:
    """Crawl một URL và trả về dict theo schema."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        title = "Unknown"
        if result.metadata and isinstance(result.metadata, dict):
            title = result.metadata.get("title", "Unknown")
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            if not article["content_markdown"].strip():
                print(f"Warning: empty content from {url}")
                continue
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
