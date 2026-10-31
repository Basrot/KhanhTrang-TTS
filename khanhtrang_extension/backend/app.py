import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from adapter import generate

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
BASE = Path(__file__).parent
AUDIO = BASE / "generated"
JOBS = BASE / "jobs"
AUDIO.mkdir(exist_ok=True)
JOBS.mkdir(exist_ok=True)
lock = threading.Lock()

def job_path(job_id):
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("Mã công việc không hợp lệ")
    return JOBS / (job_id + ".json")

def read_job(job_id):
    p = job_path(job_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def save_job(data):
    p = job_path(data["id"])
    tmp = p.with_suffix(".tmp")
    with lock:
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

def worker(job_id):
    data = read_job(job_id)
    try:
        output = AUDIO / (job_id + ".wav")
        def update_progress(percent, stage):
            data.update(progress=percent, stage=stage)
            save_job(data)
        summary, _ = generate(url=data["url"], mode="extractive", words=220, output_path=output, progress=update_progress)
        data.update(status="done", progress=100, stage="Hoàn tất", summary=summary, audio_url="/api/audio/" + job_id)
    except Exception as exc:
        app.logger.exception("Pipeline failed for %s", job_id)
        data.update(status="error", stage="Có lỗi khi tạo bản tin", error=str(exc)[-2500:])
    save_job(data)

@app.post("/api/jobs")
def start():
    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return jsonify(error="Đường dẫn bài báo không hợp lệ"), 400
    # Reuse a running or completed job for the same URL.
    for path in sorted(JOBS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("url") == url and existing.get("status") in ("running", "done"):
                if existing["status"] == "done" and not (AUDIO / (existing["id"] + ".wav")).exists():
                    continue
                return jsonify(existing), 200
        except (ValueError, OSError, KeyError):
            continue
    job_id = uuid4().hex
    data = dict(id=job_id, url=url, status="running", progress=3, stage="Đang bắt đầu…", summary="", created_at=datetime.now(timezone.utc).isoformat())
    save_job(data)
    threading.Thread(target=worker, args=(job_id,), daemon=True).start()
    return jsonify(data), 202

@app.get("/api/jobs/<job_id>")
def status(job_id):
    try:
        data = read_job(job_id)
    except ValueError:
        return jsonify(error="Mã công việc không hợp lệ"), 400
    if data is None:
        return jsonify(error="Không tìm thấy bản tin"), 404
    return jsonify(data)

@app.get("/api/audio/<job_id>")
def audio(job_id):
    try:
        job_path(job_id)
    except ValueError:
        return jsonify(error="Mã audio không hợp lệ"), 400
    path = AUDIO / (job_id + ".wav")
    if not path.is_file():
        return jsonify(error="Chưa có âm thanh"), 404
    return send_file(path, mimetype="audio/wav", conditional=True)

if __name__ == "__main__":
    # Thread remains alive even if browser tab/panel is closed, as long as backend is running.
    app.run(host="127.0.0.1", port=8765, threaded=True, debug=False)
