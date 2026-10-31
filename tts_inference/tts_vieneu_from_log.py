import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "summarizer" / "data" / "summaries_log.jsonl"
OUTPUT_DIR = ROOT / "outputs"
REF_AUDIO = ROOT / "data" / "f5tts_dataset" / "wavs" / "clip_128.wav"

MODEL_PATH = ROOT / "VieNeu-TTS" / "finetune" / "output" / "khanhtrang_v3_lora_r8" / "merged"


def slugify(text, max_len=60):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text)
    return text[:max_len].strip("-") or "ban-tin"


def read_latest_record():
    if not LOG_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy log: {LOG_PATH}")

    last_line = None
    with LOG_PATH.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                last_line = line

    if last_line is None:
        raise ValueError("File log đang rỗng.")

    return json.loads(last_line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)
    args = parser.parse_args()

    if args.text:
        text = args.text
        title = "van-ban-tu-go"
    else:
        record = read_latest_record()
        text = record["summary"]
        title = record.get("title") or "ban-tin"
        print(f"[INFO] Đọc bản tóm tắt: {title}")

    if not REF_AUDIO.is_file():
        raise FileNotFoundError(f"Không tìm thấy audio mẫu: {REF_AUDIO}")

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Không tìm thấy model AFTER: {MODEL_PATH}")

    from vieneu import Vieneu

    print("[INFO] Đang load VieNeu-TTS AFTER...", flush=True)

    tts = Vieneu(
        mode="v3turbo",
        backbone_repo=str(MODEL_PATH),
        model_subfolder="update",
        backend="pytorch",
        device="cuda"
    )

    print("[INFO] Đang tổng hợp giọng đọc...", flush=True)

    audio = tts.infer(
        text,
        ref_audio=str(REF_AUDIO),
        denoise=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / (slugify(title) + ".wav")

    tts.save(audio, str(out_path))
    print(f"[HOÀN TẤT] File audio: {out_path}", flush=True)


if __name__ == "__main__":
    main()
