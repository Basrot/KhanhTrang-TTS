"""
BƯỚC 3: TRANSCRIBE CÁC CLIP BẰNG PhoWhisper -> TẠO DATASET CHUẨN F5-TTS
==========================================================================
Input : data/processed/segments/ + segments_manifest.csv (đã tạo ở bước 2)
Output: data/f5tts_dataset/
            ├── wavs/           <- copy các clip qua đây (giữ nguyên tên)
            └── metadata.csv    <- mỗi dòng: wavs/<ten_file>.wav|<transcript>
"""

import csv
import shutil
import sys
from pathlib import Path

import torch
from transformers import pipeline

MODEL_NAME = "vinai/PhoWhisper-base"

SCRIPT_DIR = Path(__file__).parent
SEGMENTS_DIR = (SCRIPT_DIR / ".." / "data" / "processed" / "segments").resolve()
MANIFEST_CSV = (SCRIPT_DIR / ".." / "data" / "processed" / "segments_manifest.csv").resolve()

OUT_DIR = (SCRIPT_DIR / ".." / "data" / "f5tts_dataset").resolve()
OUT_WAVS_DIR = OUT_DIR / "wavs"
OUT_METADATA_CSV = OUT_DIR / "metadata.csv"

# File phụ để bạn review/sửa tay dễ hơn (có thêm cột review đã kiểm tra chưa)
REVIEW_CSV = OUT_DIR / "review_checklist.csv"


def load_asr_pipeline():
    device = 0 if torch.cuda.is_available() else -1  # transformers dùng 0 = GPU đầu tiên, -1 = CPU
    print(f"[INFO] Đang load model '{MODEL_NAME}' ({'GPU' if device == 0 else 'CPU'}) ...",
          file=sys.stderr)
    return pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
        chunk_length_s=30,  # PhoWhisper cũng giới hạn ~30s/lần, clip mình đã cắt 3-10s nên an toàn
    )


def read_manifest_rows() -> list[dict]:
    if not MANIFEST_CSV.exists():
        print(f"[LỖI] Không tìm thấy {MANIFEST_CSV}. Hãy chạy 02_segment_audio.py trước.",
              file=sys.stderr)
        sys.exit(1)
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_manifest_rows()
    if not rows:
        print("[LỖI] Manifest rỗng, không có clip nào để transcribe.", file=sys.stderr)
        return

    OUT_WAVS_DIR.mkdir(parents=True, exist_ok=True)

    asr = load_asr_pipeline()

    metadata_lines = []
    review_rows = []

    print(f"[INFO] Bắt đầu transcribe {len(rows)} clip...", file=sys.stderr)

    for i, row in enumerate(rows, start=1):
        segment_file = row["segment_file"]
        src_path = SEGMENTS_DIR / segment_file

        if not src_path.exists():
            print(f"[BỎ QUA] Không tìm thấy file {src_path}", file=sys.stderr)
            continue

        try:
            result = asr(str(src_path))
            transcript = result["text"].strip()
        except Exception as e:
            print(f"[LỖI] Transcribe thất bại cho {segment_file}: {e}", file=sys.stderr)
            continue

        if not transcript:
            print(f"[CẢNH BÁO] Transcript rỗng cho {segment_file}, bỏ qua clip này.",
                  file=sys.stderr)
            continue

        # Copy file audio sang thư mục dataset chuẩn F5-TTS
        dst_path = OUT_WAVS_DIR / segment_file
        shutil.copy2(src_path, dst_path)

        # Định dạng bắt buộc của F5-TTS: "wavs/<ten_file>.wav|<transcript>"
        metadata_lines.append(f"wavs/{segment_file}|{transcript}")

        review_rows.append({
            "segment_file": segment_file,
            "transcript": transcript,
            "duration_sec": row.get("duration_sec", ""),
            "da_kiem_tra": "chua",  # bạn tự đổi thành "roi" sau khi nghe + sửa xong
        })

        if i % 10 == 0 or i == len(rows):
            print(f"[INFO] Đã xử lý {i}/{len(rows)} clip...", file=sys.stderr)

    # Ghi metadata.csv theo đúng chuẩn F5-TTS (không có header, phân cách bằng dấu |)
    with open(OUT_METADATA_CSV, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(metadata_lines) + "\n")

    # Ghi file review riêng, dễ mở bằng Excel để nghe và sửa tay
    with open(REVIEW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["segment_file", "transcript", "duration_sec", "da_kiem_tra"]
        )
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"\n[HOÀN TẤT] Transcribe xong {len(metadata_lines)}/{len(rows)} clip.")
    print(f"  -> Dataset chuẩn F5-TTS: {OUT_DIR}")
    print(f"  -> File để review/sửa tay: {REVIEW_CSV}")
    print("\n[BẮT BUỘC] Trước khi train: mở review_checklist.csv, nghe lại từng clip trong "
          "wavs/, sửa transcript sai (đặc biệt tên riêng/số liệu), sau đó SỬA LẠI đúng "
          "nội dung tương ứng trong metadata.csv (vì metadata.csv mới là file F5-TTS thực sự đọc).")


if __name__ == "__main__":
    main()