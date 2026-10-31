"""
CHỌN CẶP REF_AUDIO / REF_TEXT CHUẨN NHẤT CHO ZERO-SHOT TTS
===============================================================
Vấn đề: F5-TTS ước lượng thời lượng audio cần sinh dựa theo tỷ lệ
(độ dài audio mẫu / độ dài text mẫu) -> nếu cặp REF_AUDIO/REF_TEXT bị lệch
(ví dụ audio ngắn nhưng text dài), model sẽ hiểu nhầm "giọng này nói rất
nhanh" và ép audio mới ra quá ngắn, nói dồn/cắt cụt.

Script này đọc toàn bộ clip đã có (manifest + transcript), tính số từ/giây
của từng clip, rồi gợi ý các clip có tốc độ hợp lý (gần với tốc độ đọc tin
tức tiếng Việt tự nhiên, khoảng 2.3-3.3 từ/giây) VÀ đủ dài (5-12 giây) để
làm audio mẫu tốt nhất.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).parent
MANIFEST_CSV = ROOT / "data" / "processed" / "segments_manifest.csv"
OLD_METADATA_CSV = ROOT / "data" / "f5tts_dataset" / "metadata_ten_cu.csv"  # tên cũ + transcript
NEW_METADATA_CSV = ROOT / "data" / "f5tts_dataset" / "metadata.csv"        # tên mới (clip_NNN.wav) + transcript

# Khoảng tốc độ nói tự nhiên của tiếng Việt khi đọc tin tức (từ/giây)
TARGET_WPS_MIN = 2.0
TARGET_WPS_MAX = 3.5
MIN_DURATION = 4.0   # audio mẫu quá ngắn -> ít thông tin về giọng
MAX_DURATION = 14.0  # audio mẫu quá dài -> chậm lúc xử lý, dễ lẫn nhiều câu


def load_duration_map() -> dict:
    """segment_file (tên cũ) -> duration_sec, lấy từ bước cắt audio ban đầu."""
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        return {row["segment_file"]: float(row["duration_sec"]) for row in csv.DictReader(f)}


def load_new_filenames_in_order() -> list:
    """Danh sách tên file MỚI (clip_001.wav, ...) theo đúng thứ tự dòng trong metadata.csv."""
    names = []
    with open(NEW_METADATA_CSV, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                rel_path = line.split("|", 1)[0]
                names.append(rel_path.replace("wavs/", "", 1))
    return names


def main():
    if not MANIFEST_CSV.exists() or not OLD_METADATA_CSV.exists():
        print("[LỖI] Thiếu file segments_manifest.csv hoặc metadata_ten_cu.csv. "
              "Hãy chắc chắn đã chạy đủ các bước trước (cắt audio, transcribe, đổi tên).")
        return

    duration_map = load_duration_map()
    new_filenames = load_new_filenames_in_order()

    candidates = []

    with open(OLD_METADATA_CSV, "r", encoding="utf-8") as f:
        old_lines = [line.rstrip("\n") for line in f if "|" in line]

    for i, line in enumerate(old_lines):
        old_rel_path, transcript = line.split("|", 1)
        old_filename = old_rel_path.replace("wavs/", "", 1)

        duration = duration_map.get(old_filename)
        if duration is None:
            continue

        word_count = len(transcript.split())
        if duration <= 0:
            continue
        wps = word_count / duration

        new_filename = new_filenames[i] if i < len(new_filenames) else "?"

        candidates.append({
            "new_filename": new_filename,
            "old_filename": old_filename,
            "duration": duration,
            "word_count": word_count,
            "wps": wps,
            "transcript": transcript,
        })

    # Lọc theo điều kiện: thời lượng hợp lý + tốc độ nói hợp lý
    good = [
        c for c in candidates
        if MIN_DURATION <= c["duration"] <= MAX_DURATION
        and TARGET_WPS_MIN <= c["wps"] <= TARGET_WPS_MAX
    ]

    # Sắp theo độ "chuẩn" (càng gần tốc độ trung tâm 2.75 từ/giây càng tốt)
    target_center = (TARGET_WPS_MIN + TARGET_WPS_MAX) / 2
    good.sort(key=lambda c: abs(c["wps"] - target_center))

    print(f"Tổng số clip có data đầy đủ: {len(candidates)}")
    print(f"Số clip đạt tiêu chuẩn (thời lượng {MIN_DURATION}-{MAX_DURATION}s, "
          f"tốc độ {TARGET_WPS_MIN}-{TARGET_WPS_MAX} từ/giây): {len(good)}\n")

    print("===== TOP 10 GỢI Ý TỐT NHẤT (dùng làm REF_AUDIO/REF_TEXT) =====\n")
    for c in good[:10]:
        print(f"File mới : {c['new_filename']}")
        print(f"Thời lượng: {c['duration']:.1f}s | Số từ: {c['word_count']} | "
              f"Tốc độ: {c['wps']:.2f} từ/giây")
        print(f"Transcript: {c['transcript']}")
        print("-" * 70)

    if not good:
        print("[CẢNH BÁO] Không có clip nào đạt chuẩn — có thể transcript hoặc "
              "thời lượng data chưa chính xác, hoặc dải tiêu chuẩn cần nới lỏng.")


if __name__ == "__main__":
    main()
