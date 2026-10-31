"""
lay url
"""

import sys
import json
from dataclasses import dataclass, asdict

import trafilatura


@dataclass
class Article:
    url: str
    title: str | None
    date: str | None
    text: str


def extract_article(url: str) -> Article | None:
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        print(f"[LỖI] Không tải được nội dung từ URL: {url}", file=sys.stderr)
        return None

    
    result_json = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        output_format="json",
        with_metadata=True,
    )

    if not result_json:
        print(f"[LỖI] Không trích xuất được nội dung chính từ: {url}", file=sys.stderr)
        return None

    data = json.loads(result_json)
    text = (data.get("text") or "").strip()

    if not text:
        print(f"[LỖI] Trích xuất ra rỗng, có thể trang này chặn bot hoặc cấu trúc lạ: {url}",
              file=sys.stderr)
        return None

    return Article(
        url=url,
        title=data.get("title"),
        date=data.get("date"),
        text=text,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python extract_article.py <URL>")
        sys.exit(1)

    article = extract_article(sys.argv[1])
    if article:
        print(f"Tiêu đề: {article.title}")
        print(f"Ngày đăng: {article.date}")
        print(f"Độ dài text: {len(article.text)} ký tự")
        print("--- Nội dung (500 ký tự đầu) ---")
        print(article.text[:500])