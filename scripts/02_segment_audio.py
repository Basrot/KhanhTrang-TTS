"""
BƯỚC 2: TÁCH FILE THÀNH CÁC CLIP NGẮN (phù hợp fine-tune F5-TTS trên GPU 8GB)
=============================================================================
Việc cần làm ở bước này (đọc input từ data/interim/resampled/, do bước 1 tạo ra):

  - Với mỗi file:
      + Nếu thời lượng file < min_duration_to_split_sec (mặc định 15s)
        -> GIỮ NGUYÊN CẢ FILE, không tách (đúng yêu cầu: file 26s thì không tách).
      + Nếu dài hơn -> phát hiện khoảng lặng (silence) để cắt tại chỗ ngắt câu
        tự nhiên, gộp các đoạn lời nói liền nhau lại thành clip có độ dài nằm
        trong khoảng [min_segment_sec, max_segment_sec] (mặc định 3-10 giây).
      + Clip nào sau khi cắt vẫn quá ngắn (< drop_shorter_than_sec) thì loại bỏ.

  - Lưu từng clip ra data/processed/segments/<ten_file>_seg{i}.wav
  - Ghi lại "bản đồ" toàn bộ clip vào data/processed/segments_manifest.csv
    (cần cho bước transcribe/tạo transcript ở giai đoạn sau)

Lưu ý: đây là cắt theo khoảng lặng (heuristic), không phải cắt hoàn hảo 100%.
Sau khi chạy xong, nên nghe thử ngẫu nhiên vài chục clip để kiểm tra trước khi
đưa vào transcribe + train.
"""

import csv
import logging
from pathlib import Path

import yaml
from pydub import AudioSegment, silence

# ---------------------------------------------------------------
# Load cấu hình
# ---------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / ".." / "configs" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

SCRIPT_DIR = Path(__file__).parent
RESAMPLED_DIR = (SCRIPT_DIR / cfg["paths"]["resampled_dir"]).resolve()
SEGMENTS_DIR = (SCRIPT_DIR / cfg["paths"]["segments_dir"]).resolve()
MANIFEST_CSV = (SCRIPT_DIR / cfg["paths"]["manifest_csv"]).resolve()
LOG_FILE = (SCRIPT_DIR / cfg["paths"]["log_file"]).resolve()

seg_cfg = cfg["segmentation"]
MIN_DURATION_TO_SPLIT_MS = seg_cfg["min_duration_to_split_sec"] * 1000
MIN_SEGMENT_MS = seg_cfg["min_segment_sec"] * 1000
MAX_SEGMENT_MS = seg_cfg["max_segment_sec"] * 1000
DROP_SHORTER_THAN_MS = seg_cfg["drop_shorter_than_sec"] * 1000
MIN_SILENCE_LEN_MS = seg_cfg["min_silence_len_ms"]
SILENCE_THRESH_OFFSET_DB = seg_cfg["silence_thresh_offset_db"]
KEEP_SILENCE_MS = seg_cfg["keep_silence_ms"]

SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("segment_audio")


def merge_nonsilent_ranges(ranges, min_len_ms, max_len_ms):
    """
    Gộp các khoảng "có tiếng nói" (nonsilent) liền nhau lại thành từng clip
    có độ dài trong khoảng [min_len_ms, max_len_ms], KHÔNG bao giờ cắt ngang
    giữa 1 đoạn có tiếng nói (chỉ cắt tại các khoảng lặng đã phát hiện).
    """
    if not ranges:
        return []

    merged = []
    cur_start, cur_end = ranges[0]

    for start, end in ranges[1:]:
        candidate_len = end - cur_start
        if candidate_len <= max_len_ms:
            # gộp thêm vẫn chưa vượt max -> gộp tiếp
            cur_end = end
        else:
            if (cur_end - cur_start) >= min_len_ms:
                # đoạn hiện tại đã đủ dài -> chốt lại, bắt đầu đoạn mới
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
            else:
                # đoạn hiện tại còn quá ngắn, buộc phải gộp thêm dù hơi vượt max
                # (chấp nhận vượt nhẹ còn hơn tạo ra clip quá ngắn)
                cur_end = end

    merged.append((cur_start, cur_end))

    # Nếu 1 đoạn tiếng nói liên tục (không có khoảng lặng) quá dài -> cắt cứng
    # theo max_len_ms (fallback, không tối ưu bằng cắt tại khoảng lặng)
    final = []
    for start, end in merged:
        length = end - start
        if length <= max_len_ms * 1.5:  # cho phép nới nhẹ trước khi phải cắt cứng
            final.append((start, end))
        else:
            cursor = start
            while cursor < end:
                chunk_end = min(cursor + int(max_len_ms), end)
                final.append((cursor, chunk_end))
                cursor = chunk_end
    return final


def segment_one_file(wav_path: Path, manifest_rows: list):
    audio = AudioSegment.from_wav(wav_path)
    duration_ms = len(audio)
    base_name = wav_path.stem

    # --- Điều kiện: file quá ngắn thì giữ nguyên, không tách ---
    if duration_ms < MIN_DURATION_TO_SPLIT_MS:
        out_name = f"{base_name}_seg0.wav"
        out_path = SEGMENTS_DIR / out_name
        audio.export(out_path, format="wav")
        manifest_rows.append({
            "source_file": wav_path.name,
            "segment_file": out_name,
            "start_ms": 0,
            "end_ms": duration_ms,
            "duration_sec": round(duration_ms / 1000.0, 2),
            "split": "no (file goc da ngan hon nguong)",
        })
        log.info(f"[GIU NGUYEN] {wav_path.name} ({duration_ms/1000:.1f}s < "
                 f"{MIN_DURATION_TO_SPLIT_MS/1000:.0f}s) -> {out_name}")
        return

    # --- File đủ dài -> phát hiện khoảng lặng để cắt ---
    silence_thresh = audio.dBFS + SILENCE_THRESH_OFFSET_DB
    nonsilent_ranges = silence.detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=silence_thresh,
    )

    if not nonsilent_ranges:
        log.warning(f"[BỎ QUA] Không phát hiện tiếng nói trong {wav_path.name} "
                    f"(có thể toàn khoảng lặng hoặc noise).")
        return

    merged_ranges = merge_nonsilent_ranges(nonsilent_ranges, MIN_SEGMENT_MS, MAX_SEGMENT_MS)

    seg_index = 0
    for start, end in merged_ranges:
        # tạo khoảng trống, tạo ngữ cảnh
        start_padded = max(0, start - KEEP_SILENCE_MS)
        end_padded = min(duration_ms, end + KEEP_SILENCE_MS)
        seg_len = end_padded - start_padded

        if seg_len < DROP_SHORTER_THAN_MS:
            continue  # Không cắt clip bé

        clip = audio[start_padded:end_padded]
        out_name = f"{base_name}_seg{seg_index}.wav"
        out_path = SEGMENTS_DIR / out_name
        clip.export(out_path, format="wav")

        manifest_rows.append({
            "source_file": wav_path.name,
            "segment_file": out_name,
            "start_ms": start_padded,
            "end_ms": end_padded,
            "duration_sec": round(seg_len / 1000.0, 2),
            "split": "yes",
        })
        seg_index += 1

    log.info(f"[TACH] {wav_path.name} ({duration_ms/1000:.1f}s) -> {seg_index} clip")


def main():
    if not RESAMPLED_DIR.exists():
        log.error(f"Chưa có thư mục {RESAMPLED_DIR}. Hãy chạy 01_extract_and_resample.py trước.")
        return

    wav_files = sorted(RESAMPLED_DIR.glob("*.wav"))
    if not wav_files:
        log.warning(f"Không có file .wav nào trong {RESAMPLED_DIR}.")
        return

    log.info(f"Bắt đầu tách {len(wav_files)} file...")

    manifest_rows = []
    for wav_path in wav_files:
        segment_one_file(wav_path, manifest_rows)

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_file", "segment_file", "start_ms", "end_ms", "duration_sec", "split"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_clips = len(manifest_rows)
    total_hours = sum(r["duration_sec"] for r in manifest_rows) / 3600.0
    log.info(f"HOÀN TẤT: {total_clips} clip, tổng {total_hours:.2f} giờ audio.")
    log.info(f"Manifest lưu tại: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()