"""
tom tat


"""

import sys
import json
import datetime
from pathlib import Path

from extract_article import extract_article
from summarize import summarize_text

LOG_PATH = Path(__file__).parent / ".." / "data" / "summaries_log.jsonl"


def run_pipeline(url: str) -> dict | None:
    article = extract_article(url)
    if article is None:
        return None

    print(f"[OK] Trích xuất thành công: '{article.title}' ({len(article.text)} ký tự)")

    summary = summarize_text(article.text)

    record = {
        "url": article.url,
        "title": article.title,
        "date": article.date,
        "text": article.text,
        "summary": summary,
        "processed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python pipeline.py <URL bài báo>")
        sys.exit(1)

    url = sys.argv[1]
    result = run_pipeline(url)

    if result:
        print("\n================ TÓM TẮT ================")
        print(f"Tiêu đề : {result['title']}")
        print(f"Ngày    : {result['date']}")
        print("-------------------------------------------")
        print(result["summary"])
        print("===========================================")
        print(f"\n(Đã lưu vào {LOG_PATH.resolve()})")