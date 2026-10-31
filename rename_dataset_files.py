"""
ĐỔI TÊN HÀNG LOẠT FILE TRONG wavs/ SANG TÊN ĐƠN GIẢN, AN TOÀN
==================================================================
Tên file gốc (lấy theo tiêu đề video YouTube) có dấu #, khoảng trắng, tiếng
Việt có dấu -> dễ gây lỗi khi dùng trong code/đường dẫn (như bạn vừa gặp).

Script này:
  1. Đọc data/f5tts_dataset/metadata.csv (định dạng: wavs/<ten_file>.wav|<transcript>)
  2. Đổi tên từng file .wav sang dạng clip_001.wav, clip_002.wav, ...
  3. Ghi lại metadata.csv mới với tên file đã đổi, transcript giữ nguyên

CHẠY 1 LẦN DUY NHẤT. Sau khi chạy xong, dùng tên file mới (rất ngắn gọn,
không dấu, không ký tự đặc biệt) cho REF_AUDIO và mọi chỗ khác.
"""

import csv
import shutil
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "data" / "f5tts_dataset"
WAVS_DIR = DATASET_DIR / "wavs"
METADATA_CSV = DATASET_DIR / "metadata.csv"
BACKUP_METADATA_CSV = DATASET_DIR / "metadata_ten_cu.csv"


def main():
    if not METADATA_CSV.exists():
        print(f"[LỖI] Không tìm thấy {METADATA_CSV}")
        return

    # Backup lại metadata gốc, phòng khi cần đối chiếu
    shutil.copy2(METADATA_CSV, BACKUP_METADATA_CSV)
    print(f"[INFO] Đã backup metadata gốc -> {BACKUP_METADATA_CSV}")

    with open(METADATA_CSV, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    new_lines = []
    renamed_count = 0

    for i, line in enumerate(lines, start=1):
        if "|" not in line:
            print(f"[BỎ QUA] Dòng không đúng định dạng: {line[:50]}...")
            continue

        old_rel_path, transcript = line.split("|", 1)  # vd: "wavs/ten_cu.wav"
        old_filename = old_rel_path.replace("wavs/", "", 1)
        old_path = WAVS_DIR / old_filename

        if not old_path.exists():
            print(f"[CẢNH BÁO] Không tìm thấy file: {old_path}")
            continue

        new_filename = f"clip_{i:03d}.wav"
        new_path = WAVS_DIR / new_filename

        old_path.rename(new_path)
        renamed_count += 1

        new_lines.append(f"wavs/{new_filename}|{transcript}")

    with open(METADATA_CSV, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"\n[HOÀN TẤT] Đã đổi tên {renamed_count} file.")
    print(f"Metadata mới (tên gọn, an toàn) đã ghi vào: {METADATA_CSV}")
    print(f"Metadata cũ (để đối chiếu nếu cần) lưu tại: {BACKUP_METADATA_CSV}")
    print(f"\nVí dụ: REF_AUDIO giờ có thể dùng dạng:")
    print(rf'  REF_AUDIO = r"{WAVS_DIR}\clip_001.wav"')


if __name__ == "__main__":
    main()
