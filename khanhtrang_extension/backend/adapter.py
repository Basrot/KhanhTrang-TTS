"""Run the user's existing PowerShell URL -> bulletin -> VieNeu pipeline."""
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'summarizer' / 'data' / 'summaries_log.jsonl'
OUT = ROOT / 'outputs'
LOCK = threading.Lock()  # the TTS script always reads the latest log record


def generate(url: str, mode: str, words: int, output_path: Path, progress=None) -> tuple[str, Path]:
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('URL bài báo phải bắt đầu bằng http:// hoặc https://')
    if mode not in ('extractive', 'llm'):
        raise ValueError('Chế độ tóm tắt không hợp lệ.')
    if not 50 <= words <= 1000:
        raise ValueError('Số từ phải nằm trong khoảng 50–1000.')
    script = ROOT / 'run_all_v2.ps1'
    if not script.is_file():
        raise FileNotFoundError(f'Không tìm thấy {script}')
    with LOCK:
        OUT.mkdir(exist_ok=True)
        before = {p.resolve(): p.stat().st_mtime_ns for p in OUT.glob('*.wav')}
        log_size = LOG.stat().st_size if LOG.exists() else 0
        # -File and separate args avoid shell string interpolation of arbitrary URLs.
        cmd = ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(script), '-Url', url, '-Mode', mode, '-Words', str(words)]
        env = os.environ.copy()
        env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        if progress:
            progress(8, 'Đang lấy nội dung bài báo…')
        lines = []
        start_time = time.monotonic()
        try:
            # Read output continuously to identify real pipeline stages.
            with subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                  errors='replace', env=env, bufsize=1,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as proc:
                for line in proc.stdout:
                    lines.append(line)
                    if len(lines) > 300:
                        lines.pop(0)
                    low = line.lower()
                    if progress:
                        if 'đã lưu' in low or 'da luu' in low or 'bước 2' in low or 'buoc 2' in low:
                            progress(45, 'Đã tóm tắt. Đang chuẩn bị giọng đọc…')
                        elif 'load vieneu' in low or 'đang load vieneu' in low:
                            progress(55, 'Đang tải mô hình giọng đọc…')
                        elif 'đang đọc' in low or 'dang doc' in low or 'infer' in low:
                            progress(70, 'Đang tổng hợp âm thanh…')
                        elif 'hoàn tất' in low or 'hoan tat' in low or 'file audio:' in low:
                            progress(94, 'Đang lưu file âm thanh…')
                    if time.monotonic() - start_time > 900:
                        proc.kill()
                        raise RuntimeError('Pipeline chạy quá 15 phút.')
                code = proc.wait()
            if code != 0:
                raise RuntimeError('Pipeline thất bại:\n' + ''.join(lines)[-2500:])
        except OSError as exc:
            raise RuntimeError('Không thể khởi chạy PowerShell: ' + str(exc)) from exc
        if not LOG.exists() or LOG.stat().st_size <= log_size:
            raise RuntimeError('Pipeline không ghi thêm bản tóm tắt mới vào summaries_log.jsonl.')
        with LOG.open('r', encoding='utf-8-sig') as f:
            records = [line.strip() for line in f if line.strip()]
        if not records:
            raise RuntimeError('File log không có bản tóm tắt.')
        record = json.loads(records[-1])
        summary = record.get('summary', '')
        if not isinstance(summary, str) or not summary.strip():
            raise RuntimeError('Không đọc được trường summary từ log.')
        changed = [p for p in OUT.glob('*.wav') if p.resolve() not in before or p.stat().st_mtime_ns != before[p.resolve()]]
        if not changed:
            raise RuntimeError('TTS chưa tạo hoặc cập nhật file WAV trong outputs.')
        newest = max(changed, key=lambda p: p.stat().st_mtime_ns)
        import shutil
        if progress:
            progress(97, 'Đang hoàn tất bản ghi âm…')
        shutil.copyfile(newest, output_path)
        return summary, output_path
