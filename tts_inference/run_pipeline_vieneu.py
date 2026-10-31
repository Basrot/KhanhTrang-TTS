"""
PIPELINE MỚI: TÓM TẮT -> ĐỌC BẰNG VieNeu-TTS (xử lý Anh-Việt tự nhiên, không cần phiên âm tay)
==================================================================================================
Thay thế hoàn toàn phần zero-shot TTS cũ (F5-TTS-Vietnamese + normalize_for_tts.py).
VieNeu-TTS tự xử lý code-switching Việt-Anh ngay trong model, nên KHÔNG CẦN import
normalize_for_tts/_v2 nữa — bỏ hẳn bước chuẩn hoá thủ công.

Cách dùng:
    python run_pipeline_vieneu.py "https://dantri.com.vn/duong-dan-bai-bao.htm"
"""

import argparse
import re
import sys
from pathlib import Path

from vieneu import Vieneu

# --- Trỏ tới code Phần 1 (tóm tắt) đã có sẵn ---
ROOT = Path(__file__).parent.parent  # .../khanhtrang-tts
sys.path.append(str(ROOT / "summarizer" / "scripts"))

from extract_article import extract_article
from summarize import summarize_text

# ==== CHỈNH LẠI CHO ĐÚNG MÁY BẠN ====
# Clip Khánh Trang làm giọng mẫu — CHỈ CẦN audio, KHÔNG CẦN transcript nữa!
REF_AUDIO = str(ROOT / "data" / "f5tts_dataset" / "wavs" / "clip_128.wav")
# =====================================

OUTPUT_DIR = ROOT / "outputs"

_tts = None


def get_tts():
    global _tts
    if _tts is None:
        print("[INFO] Đang load VieNeu-TTS (v3 Turbo, chạy CPU qua ONNX)...", file=sys.stderr)
        _tts = Vieneu()  # mặc định v3 Turbo
    return _tts


def slugify(text: str, max_len: int = 60) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
                  r"ùúụủũưừứựửữỳýỵỷỹđ\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text[:max_len].strip("-") or "bai-bao"


def run(url: str):
    print(f"[1/3] Đang lấy nội dung từ: {url}")
    article = extract_article(url)
    if article is None:
        sys.exit(1)
    print(f"      -> '{article.title}' ({len(article.text)} ký tự)")

    print("[2/3] Đang tóm tắt...")
    summary = summarize_text(article.text)
    print(f"      -> {summary}")

    print("[3/3] Đang tổng hợp giọng đọc (VieNeu-TTS, tự xử lý Anh-Việt)...")
    tts = get_tts()

    # Chỉ cần đưa summary GỐC thẳng vào — model tự đọc đúng cả từ tiếng Anh
    # lẫn tiếng Việt trong cùng câu, không cần qua bước chuẩn hoá nào cả.
    audio = tts.infer(summary, ref_audio=REF_AUDIO, denoise=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / (slugify(article.title or "bai-bao") + ".wav")
    tts.save(audio, str(out_path))

    print(f"\n[HOÀN TẤT] File audio: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str)
    args = parser.parse_args()
    run(args.url)
