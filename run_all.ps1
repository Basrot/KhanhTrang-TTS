
param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [ValidateSet("extractive", "llm")]
    [string]$Mode = "extractive",

    [int]$Words = 220
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$summaryPython = Join-Path $root "venv\Scripts\python.exe"
$summaryScript = Join-Path $root "summarizer\scripts\bulletin.py"

$f5Python = Join-Path $root "F5-TTS-Vietnamese\.venv\Scripts\python.exe"
$f5Script = Join-Path $root "tts_inference\tts_f5_from_log.py"

Write-Host "`n=== BƯỚC 1: Tạo bản tin ($Mode) ===" -ForegroundColor Cyan

if (-not (Test-Path $summaryPython)) {
    throw "Không tìm thấy Python của summarizer: $summaryPython"
}

& $summaryPython $summaryScript $Url --mode $Mode --words $Words

if ($LASTEXITCODE -ne 0) {
    Write-Host "Tóm tắt thất bại, dừng lại." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== BƯỚC 2: Đọc bằng F5-TTS ===" -ForegroundColor Cyan

if (-not (Test-Path $f5Python)) {
    throw "Không tìm thấy môi trường F5-TTS: $f5Python"
}

if (-not (Test-Path $f5Script)) {
    throw "Chưa có script inference F5-TTS: $f5Script"
}

$env:PYTHONPATH = Join-Path $root "F5-TTS-Vietnamese\src"

& $f5Python $f5Script

if ($LASTEXITCODE -ne 0) {
    Write-Host "F5-TTS chạy thất bại." -ForegroundColor Red
    exit 1
}

Write-Host "`nHOÀN THÀNH BẢN TIN F5-TTS!" -ForegroundColor Green
