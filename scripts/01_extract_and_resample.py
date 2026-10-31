"""
BƯỚC 1: CHUẨN HOÁ AUDIO GỐC
============================
Việc cần làm ở bước này:
  - Đọc toàn bộ 51 file gốc (mp3/mp4/wav/...) trong data/raw/
  - Nếu là file video -> tự tách lấy track audio (pydub dùng ffmpeg nền)
  - Resample về 24kHz, mono, 16-bit PCM (đúng chuẩn input của F5-TTS)
  - Peak-normalize âm lượng về khoảng -1 dBFS để các file đồng đều nhau
  - Lưu kết quả sang data/interim/resampled/<ten_file>.wav
 
Yêu cầu hệ thống: đã cài ffmpeg (pydub gọi ngầm qua ffmpeg để decode mp3/mp4/...)
    Ubuntu/Debian: sudo apt install ffmpeg
"""
 
import logging
from pathlib import Path
 
import yaml
from pydub import AudioSegment, effects
 
# ---------------------------------------------------------------
# Load cấu hình
# ---------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / ".." / "configs" / "config.yaml"
 
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
 
SCRIPT_DIR = Path(__file__).parent
RAW_DIR = (SCRIPT_DIR / cfg["paths"]["raw_dir"]).resolve()
RESAMPLED_DIR = (SCRIPT_DIR / cfg["paths"]["resampled_dir"]).resolve()
LOG_FILE = (SCRIPT_DIR / cfg["paths"]["log_file"]).resolve()
 
SAMPLE_RATE = cfg["audio"]["sample_rate"]
CHANNELS = cfg["audio"]["channels"]
SAMPLE_WIDTH = cfg["audio"]["sample_width"]
ACCEPTED_EXT = set(e.lower() for e in cfg["accepted_extensions"])
 
RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("extract_and_resample")
 
 
def peak_normalize(audio: AudioSegment, target_peak_dbfs: float = -1.0) -> AudioSegment:
    """Chuẩn hoá biên độ đỉnh về target_peak_dbfs (mặc định -1 dBFS)."""
    normalized = effects.normalize(audio)  # đưa đỉnh về 0 dBFS
    gain_to_apply = target_peak_dbfs - 0.0
    return normalized.apply_gain(gain_to_apply)
 
 
def process_one_file(src_path: Path) -> Path | None:
    try:
        audio = AudioSegment.from_file(src_path)  # pydub tự nhận diện định dạng qua ffmpeg
    except Exception as e:
        log.error(f"[BỎ QUA] Không đọc được file '{src_path.name}': {e}")
        return None
 
    duration_sec = len(audio) / 1000.0
    if duration_sec < 0.5:
        log.warning(f"[BỎ QUA] File '{src_path.name}' quá ngắn ({duration_sec:.2f}s), có thể lỗi.")
        return None
 
    audio = audio.set_channels(CHANNELS)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_sample_width(SAMPLE_WIDTH)
    audio = peak_normalize(audio, target_peak_dbfs=-1.0)
 
    out_name = src_path.stem + ".wav"
    out_path = RESAMPLED_DIR / out_name
    audio.export(out_path, format="wav")
 
    log.info(f"[OK] {src_path.name} -> {out_name} | {duration_sec:.1f}s | {SAMPLE_RATE}Hz mono")
    return out_path
 
 
def main():
    if not RAW_DIR.exists():
        log.error(f"Không tìm thấy thư mục raw: {RAW_DIR}. "
                  f"Hãy copy 51 file gốc của Khánh Trang vào đây trước.")
        return
 
    raw_files = sorted(
        p for p in RAW_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ACCEPTED_EXT
    )
 
    if not raw_files:
        log.warning(f"Không có file hợp lệ nào trong {RAW_DIR} "
                    f"(các đuôi chấp nhận: {sorted(ACCEPTED_EXT)})")
        return
 
    log.info(f"Tìm thấy {len(raw_files)} file trong data/raw/. Bắt đầu chuẩn hoá...")
 
    ok_count = 0
    for src_path in raw_files:
        result = process_one_file(src_path)
        if result is not None:
            ok_count += 1
 
    log.info(f"HOÀN TẤT: {ok_count}/{len(raw_files)} file đã được chuẩn hoá -> {RESAMPLED_DIR}")
 
 
if __name__ == "__main__":
    main()
 