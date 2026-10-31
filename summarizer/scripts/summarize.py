"""
TÓM TẮT VĂN BẢN TIẾNG VIỆT BẰNG ViT5 
"""

import sys
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "VietAI/vit5-large-vietnews-summarization"
MAX_INPUT_TOKENS = 1024
MAX_SUMMARY_TOKENS = 256
MIN_SUMMARY_TOKENS = 80   # MỚI: ép tóm tắt dài tối thiểu ~80 token (~55-65 từ tiếng Việt)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"[INFO] Đang load model '{MODEL_NAME}' lên {_device} ...", file=sys.stderr)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
    return _tokenizer, _model


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[\.\?\!])\s+", text)
    return [s for s in sentences if s.strip()]


def _chunk_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    sentences = _split_sentences(text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        n_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        if current and current_len + n_tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += n_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _generate_summary(text: str, tokenizer, model, max_summary_tokens: int,
                       min_summary_tokens: int = 0) -> str:
    input_text = text.strip() + " </s>"
    encoding = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS,
    ).to(_device)

    # Không ép min_length lớn hơn khả năng input cho phép (tránh lỗi với input rất ngắn)
    effective_min = min(min_summary_tokens, max(max_summary_tokens - 10, 0))

    with torch.no_grad():
        output_ids = model.generate(
    input_ids=encoding["input_ids"],
    attention_mask=encoding["attention_mask"],
    max_length=max_summary_tokens,
    min_length=120,              # tăng từ 80 lên 120
    num_beams=6,                 # tăng từ 4 lên 6 — duyệt nhiều phương án hơn, chất lượng cao hơn (đổi lại chậm hơn 1 chút)
    length_penalty=2.0,          # tăng từ 1.5 lên 2.0 — ưu tiên câu dài, đầy đủ ý hơn nữa
    repetition_penalty=1.3,      # MỚI — phạt việc lặp lại cụm từ/ý đã nói
    no_repeat_ngram_size=3,
    early_stopping=True,
)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def summarize_text(text: str, max_summary_tokens: int = MAX_SUMMARY_TOKENS) -> str:
    tokenizer, model = _load_model()
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    if n_tokens <= MAX_INPUT_TOKENS:
        return _generate_summary(
            text, tokenizer, model,
            max_summary_tokens=max_summary_tokens,
            min_summary_tokens=MIN_SUMMARY_TOKENS,
        )

    print(f"[INFO] Bài dài {n_tokens} token > {MAX_INPUT_TOKENS}, chia chunk để xử lý.",
          file=sys.stderr)
    chunks = _chunk_by_tokens(text, tokenizer, MAX_INPUT_TOKENS)

    partial_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"[INFO] Tóm tắt chunk {i + 1}/{len(chunks)} ...", file=sys.stderr)
        partial_summaries.append(
            _generate_summary(chunk, tokenizer, model,
                               max_summary_tokens=128, min_summary_tokens=40)
        )

    combined = " ".join(partial_summaries)
    combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))
    if combined_tokens > max_summary_tokens:
        return _generate_summary(
            combined, tokenizer, model,
            max_summary_tokens=max_summary_tokens,
            min_summary_tokens=MIN_SUMMARY_TOKENS,
        )
    return combined


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python summarize.py \"<văn bản cần tóm tắt>\"")
        sys.exit(1)
    result = summarize_text(sys.argv[1])
    print("\n--- TÓM TẮT ---")
    print(result)