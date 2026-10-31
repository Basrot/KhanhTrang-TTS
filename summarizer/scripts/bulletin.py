"""
TẠO BẢN TIN ĐỂ ĐỌC TỪ BÀI BÁO (thay cho tóm tắt ViT5)

"""

import argparse
import datetime
import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "summaries_log.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"
WORDS_PER_SECOND = 2.7  # tốc độ đọc tin tức trung bình, chỉ dùng để ước lượng thời lượng

# Tên đọc lên của các báo (bổ sung dòng mới nếu dùng báo khác)
PUBLISHERS = {
  
}

STOPWORDS = set(
    "và của là có cho các những một này được trong với để khi đã sẽ đang cũng như "
    "từ tại về ra vào theo rằng thì mà nhưng nên còn hay hoặc bị do sau trước đến "
    "nếu vì rất đều cả nhiều ông bà anh chị họ chúng tôi ta không chưa vẫn lại lên "
    "xuống qua sang đó đây ấy nào gì ai".split()
)

SKIP_PREFIXES = ("ảnh:", "ảnh minh họa", "video:", "xem thêm", "đọc thêm", "nguồn:",
                 "tin liên quan", "bài liên quan", "clip:")
AGENCY_PREFIX = re.compile(r"^\(\s*[^)]{2,30}\)\s*[-–—:]\s*")
LIST_NUMBER_PREFIX = re.compile(r"^\d{1,2}\s*[/.)]\s+")


# ---------------------------------------------------------------- tiện ích
def publisher_from_url(url: str):
    host = urlparse(url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    for domain, name in PUBLISHERS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return None


def spoken_date(date_str):
    """'2026-09-12' hoặc '12/09/2026' -> '12 tháng 9 năm 2026'."""
    if not date_str:
        return None
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", date_str)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str)
        if not m:
            return None
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        datetime.date(y, mo, d)
    except ValueError:
        return None
    return f"{d} tháng {mo} năm {y}"


def word_count(text: str) -> int:
    return len(text.split())


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def clean_paragraphs(text: str, title):
    """Giữ lại các đoạn là câu văn thật; bỏ tiêu đề lặp, chú thích ảnh, đề mục."""
    out = []
    for p in (line.strip() for line in text.split("\n")):
        if not p:
            continue
        if title and _norm(p) == _norm(title):
            continue
        low = p.lower()
        if low.startswith(SKIP_PREFIXES) or "trân trọng đăng tải" in low:
            continue
        p = AGENCY_PREFIX.sub("", p).strip()
        p = LIST_NUMBER_PREFIX.sub("", p).strip()
        if len(p) < 30:
            continue
        if p[-1] not in '.?!"”…':  # không kết thúc bằng dấu câu -> đề mục/chú thích
            continue
        out.append(p)
    return out


def split_sentences(text: str):
    pieces = re.split(r"(?<=[.!?…])\s+", text.strip())
    sents = []
    for p in pieces:
        if not p:
            continue
        if sents and (p[0].islower() or p[0].isdigit()):  # vd "TP. Hồ Chí Minh", "10/9. 2 người"
            sents[-1] += " " + p
        else:
            sents.append(p)
    return [s.strip() for s in sents if s.strip()]


# ---------------------------------------------------------------- extractive
def _content_tokens(s: str):
    return {w for w in re.findall(r"\w+", s.lower()) if w not in STOPWORDS and len(w) > 1}


def _similarity(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    return len(a & b) / (math.log(len(a)) + math.log(len(b)))


def textrank_scores(sentences, damping=0.85, iterations=30):
    n = len(sentences)
    toks = [_content_tokens(s) for s in sentences]
    w = [[_similarity(toks[i], toks[j]) if i != j else 0.0 for j in range(n)] for i in range(n)]
    row_sum = [sum(row) for row in w]
    scores = [1.0] * n
    for _ in range(iterations):
        new = []
        for i in range(n):
            acc = sum(w[j][i] / row_sum[j] * scores[j] for j in range(n) if row_sum[j] > 0)
            new.append((1 - damping) + damping * acc)
        scores = new
    return scores


def extractive_body(paragraphs, max_words=220, lead_sentences=2, lead_max_words=70):
    """Sapo (đoạn đầu) giữ nguyên + các câu quan trọng nhất của phần còn lại, theo thứ tự gốc."""
    if not paragraphs:
        return ""

    first = split_sentences(paragraphs[0])
    lead, used = [], 0
    for s in first[:lead_sentences]:
        if lead and used + word_count(s) > lead_max_words:
            break
        lead.append(s)
        used += word_count(s)

    rest = first[len(lead):]
    for p in paragraphs[1:]:
        rest.extend(split_sentences(p))

    budget = max_words - used
    if budget <= 0 or not rest:
        return " ".join(lead)

    lead_tokens = _content_tokens(" ".join(lead))
    scores = textrank_scores(rest)
    ranked = []
    for idx, (s, sc) in enumerate(zip(rest, scores)):
        wc = word_count(s)
        if wc < 7 or wc > 60:
            continue
        bonus = (1.15 if re.search(r"\d", s) else 1.0) * (1 + 0.6 * math.exp(-idx / 4))
        s_tokens = _content_tokens(s)
        overlap = len(s_tokens & lead_tokens) / max(len(s_tokens), 1)
        if overlap > 0.7:  # gần như lặp lại sapo
            continue
        ranked.append((sc * bonus, idx, s, wc))

    chosen = []
    for _, idx, s, wc in sorted(ranked, reverse=True):
        if wc <= budget:
            chosen.append((idx, s))
            budget -= wc
        if budget < 7:
            break

    chosen.sort()
    return " ".join(lead + [s for _, s in chosen])


# ---------------------------------------------------------------- LLM local
def build_prompt(title, date_text, article_text, words):
    return (
        f"Hãy viết lại bài báo dưới đây thành bản tin để đọc lên trên đài, khoảng {words} từ.\n\n"
        "Yêu cầu bắt buộc:\n"
        "- Câu đầu nêu sự kiện chính: ai, làm gì, ở đâu, khi nào.\n"
        "- Sau đó nêu các chi tiết cụ thể quan trọng: số liệu, tên người, địa danh, tổ chức, "
        "các diễn biến chính, đúng như trong bài.\n"
        "- Giữ nguyên chính xác tên riêng, chức danh, số liệu, ngày tháng. TUYỆT ĐỐI không thêm "
        "thông tin, không suy đoán, không bình luận ngoài bài.\n"
        "- Viết thành các câu văn liền mạch, dễ đọc thành tiếng. Không dùng gạch đầu dòng, "
        "tiêu đề, ký hiệu markdown hay emoji.\n"
        "- Chỉ trả về nội dung bản tin, không giải thích thêm.\n\n"
        f"Tiêu đề: {title}\n"
        f"Ngày đăng: {date_text or 'không rõ'}\n\n"
        f"Nội dung bài báo:\n<<<\n{article_text}\n>>>\n/no_think"
    )


def clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"[*#`_>]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def llm_body(title, date_text, article_text, words, model):
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": "Bạn là biên tập viên bản tin phát thanh tiếng Việt. "
                                          "Chỉ dùng thông tin có trong bài báo được cung cấp."},
            {"role": "user", "content": build_prompt(title, date_text, article_text[:8000], words)},
        ],
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 900},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return clean_llm_output(data["message"]["content"])


# ---------------------------------------------------------------- kiểm tra bịa thông tin
def check_faithfulness(summary: str, source: str):
    """Trả về (số_lạ, tên_lạ): số/tên riêng xuất hiện trong bản tin nhưng KHÔNG có trong bài gốc."""
    src_lower = source.lower()
    src_nums = set(re.findall(r"\d+", source))
    src_words = set(re.findall(r"\w+", src_lower))

    bad_numbers = sorted({n for n in re.findall(r"\d+", summary) if n not in src_nums})

    bad_names = []
    for sent in split_sentences(summary):
        toks = re.findall(r"\w+", sent)[1:]  # bỏ từ đầu câu (luôn viết hoa)
        run = []
        for tok in toks + [""]:
            if tok and tok[0].isupper() and not tok.isdigit():
                run.append(tok)
                # từ viết tắt/CamelCase (UAV, OpenAI): kiểm tra riêng từng từ
                if sum(c.isupper() for c in tok) >= 2 and tok.lower() not in src_words:
                    bad_names.append(tok)
            else:
                if len(run) >= 2:
                    phrase = " ".join(run)
                    new_word = any(w.lower() not in src_words for w in run)
                    if phrase.lower() not in src_lower and new_word:
                        bad_names.append(phrase)
                run = []
    return bad_numbers, sorted(set(bad_names))


# ---------------------------------------------------------------- ghép bản tin
def compose_bulletin(url, title, date_str, body):
    parts = []
    publisher = publisher_from_url(url)
    d = spoken_date(date_str)
    if publisher and d:
        parts.append(f"Bản tin từ {publisher}, đăng ngày {d}.")
    elif publisher:
        parts.append(f"Bản tin từ {publisher}.")
    elif d:
        parts.append(f"Bản tin đăng ngày {d}.")
    if title:
        parts.append(title.strip().rstrip(".!?:;,") + ".")
    parts.append(body.strip())
    return " ".join(parts)


def build_bulletin(article, mode="extractive", words=220, model=DEFAULT_MODEL):
    """Trả về (bản_tin, chế_độ_đã_dùng, ghi_chú)."""
    paragraphs = clean_paragraphs(article.text, article.title)
    if not paragraphs:  # trang lạ không lọc được -> dùng nguyên văn
        paragraphs = [p.strip() for p in article.text.split("\n") if len(p.strip()) > 30]

    note = ""
    used_mode = "extractive"
    body = ""

    if mode == "llm":
        try:
            print(f"[INFO] Đang nhờ model local '{model}' viết bản tin...", file=sys.stderr)
            candidate = llm_body(article.title, spoken_date(article.date), article.text, words, model)
            bad_nums, bad_names = check_faithfulness(candidate, article.text)
            if not candidate:
                note = "LLM trả về rỗng -> dùng extractive."
            elif bad_nums or len(bad_names) >= 2:
                note = f"LLM bị nghi bịa (số lạ: {bad_nums}, tên lạ: {bad_names}) -> dùng extractive."
            else:
                body, used_mode = candidate, "llm"
                if bad_names:
                    note = f"Lưu ý nhẹ, tên chưa thấy trong bài gốc: {bad_names}"
        except (urllib.error.URLError, ConnectionError, TimeoutError, KeyError) as e:
            note = f"Không gọi được Ollama ({e}) -> dùng extractive."

    if not body:
        body = extractive_body(paragraphs, max_words=words)

    return compose_bulletin(article.url, article.title, article.date, body), used_mode, note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--mode", choices=["extractive", "llm"], default="extractive")
    parser.add_argument("--words", type=int, default=220, help="Số từ mục tiêu cho phần nội dung")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Tên model Ollama (chế độ llm)")
    args = parser.parse_args()

    from extract_article import extract_article  # import trễ: cần trafilatura

    article = extract_article(args.url)
    if article is None:
        sys.exit(1)
    print(f"[OK] '{article.title}' | ngày {article.date} | {len(article.text)} ký tự")

    bulletin, used_mode, note = build_bulletin(article, args.mode, args.words, args.model)
    if note:
        print(f"[CHÚ Ý] {note}")

    record = {
        "url": article.url, "title": article.title, "date": article.date,
        "text": article.text, "summary": bulletin, "mode": used_mode,
        "processed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = word_count(bulletin)
    print(f"\n=========== BẢN TIN ({used_mode}) ===========")
    print(bulletin)
    print(f"=============================================")
    print(f"{n} từ, ước lượng đọc khoảng {n / WORDS_PER_SECOND:.0f} giây. Đã lưu vào {LOG_PATH}")


if __name__ == "__main__":
    main()
