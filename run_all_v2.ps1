
param(
    [Parameter(Mandatory = $true)][string]$Url,
    [ValidateSet("extractive", "llm")][string]$Mode = "extractive",
    [int]$Words = 220
)

$root = $PSScriptRoot

Write-Host "=== BƯỚC 1: Tạo bản tin (chế độ $Mode) ===" -ForegroundColor Cyan
& "$root\venv\Scripts\python.exe" "$root\summarizer\scripts\bulletin.py" $Url --mode $Mode --words $Words
if ($LASTEXITCODE -ne 0) {
    Write-Host "Bước tạo bản tin lỗi, dừng lại." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== BƯỚC 2: Đọc bằng giọng Khánh Trang (venv_tts) ===" -ForegroundColor Cyan
$env:PYTHONPATH = "C:\Khanhtrang-tts\VieNeu-TTS\src"
& ".\VieNeu-TTS\.venv\Scripts\python.exe" "tts_inference\tts_vieneu_from_log.py"
