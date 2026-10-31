#!/usr/bin/env python3
"""Independent ASR audit of a Vietnamese WAV/transcript dataset.
Read-only source files. Outputs are suggestions, not ground truth.
"""
import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


def norm(s):
    s = unicodedata.normalize('NFC', s or '').lower()
    s = re.sub(r'[^\w\s]', ' ', s, flags=re.UNICODE)
    return ' '.join(s.split())


def read_metadata(p):
    rows = []
    for i, line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        if '|' not in line:
            raise ValueError(f'Malformed metadata line {i}: {line[:80]}')
        rel, txt = line.split('|', 1)
        name = Path(rel.replace('\\', '/')).name
        if not name or not txt.strip():
            raise ValueError(f'Empty filename/transcript on line {i}')
        rows.append((name, txt.strip()))
    if len({n for n, _ in rows}) != len(rows):
        raise ValueError('Duplicate WAV names in metadata')
    return rows


def read_review(p):
    if not p.exists():
        return {}
    with p.open(encoding='utf-8-sig', newline='') as f:
        return {Path(r['clip'].replace('\\', '/')).name: r for r in csv.DictReader(f)}


def read_test(p):
    if not p.exists():
        return set()
    with p.open(encoding='utf-8-sig', newline='') as f:
        return {Path(r['wav'].replace('\\', '/')).name for r in csv.DictReader(f)}


def classify(original, second, flags, threshold):
    a, b = norm(original), norm(second)
    similarity = SequenceMatcher(None, a, b).ratio() if a and b else 0.0
    suspicious = bool(re.search(r'\b(?:unk|unknown|<unk>)\b', original, re.I))
    if not b:
        return 'needs_review', similarity, 'ASR2_empty'
    if similarity >= threshold and not suspicious and not flags:
        return 'low_disagreement', similarity, ''
    reasons = []
    if similarity < threshold:
        reasons.append('ASR_disagreement')
    if suspicious:
        reasons.append('unknown_token')
    if flags:
        reasons.append('previous_audit_flag')
    return 'needs_review', similarity, ';'.join(reasons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(r'C:\Khanhtrang-tts'))
    ap.add_argument('--model', default='small', help='faster-whisper model: small, medium, large-v3, etc.')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--compute-type', default='auto', help='auto, int8, float16, int8_float16')
    ap.add_argument('--threshold', type=float, default=0.88)
    ap.add_argument('--scope', choices=['all', 'development'], default='all')
    ap.add_argument('--limit', type=int, default=0, help='0 = all; use 10 for smoke test')
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    if not 0 <= args.threshold <= 1:
        ap.error('--threshold must be between 0 and 1')
    root = args.root.resolve()
    ds = root / 'data' / 'f5tts_dataset'
    rows = read_metadata(ds / 'metadata.csv')
    reviews = read_review(root / 'eval' / 'dataset_review' / 'reviewed_dataset.csv')
    test_ids = read_test(root / 'eval' / 'data' / 'test.csv')
    if args.scope == 'development':
        old = read_metadata(ds / 'metadata_ten_cu.csv')
        manifest_path = root / 'data' / 'processed' / 'segments_manifest.csv'
        with manifest_path.open(encoding='utf-8-sig', newline='') as f:
            manifest = list(csv.DictReader(f))
        if len(rows) != len(old) or len(rows) != len(manifest):
            raise ValueError('Cannot safely map source videos: metadata/manifest lengths differ')
        seg2src = {r['segment_file']: r['source_file'] for r in manifest}
        mapping = {}
        for (new_name, _), (old_name, _) in zip(rows, old):
            if old_name not in seg2src:
                raise ValueError(f'Cannot map original segment: {old_name}')
            mapping[new_name] = seg2src[old_name]
        if not test_ids <= mapping.keys():
            raise ValueError('Test WAV not in source mapping')
        test_sources = {mapping[n] for n in test_ids}
        rows = [(n, t) for n, t in rows if mapping[n] not in test_sources]
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('No WAV rows selected')
    output = args.output or root / 'eval' / 'dataset_audit' / f'asr_second_opinion_{args.scope}.csv'
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output == (ds / 'metadata.csv').resolve():
        raise ValueError('Refusing to overwrite original metadata')
    if output.exists():
        raise FileExistsError(f'Output exists: {output}. Choose another --output to preserve earlier results.')
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit('Missing dependency. Install with: python -m pip install faster-whisper')
    print(f'Loading faster-whisper model={args.model}, device={args.device}, compute={args.compute_type}', flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    fields = ['clip','is_fixed_test','review_status','audit_flags','original_text','asr2_text',
              'similarity','category','reasons','suggested_text','error']
    counts = {'low_disagreement': 0, 'needs_review': 0, 'error': 0}
    start = time.time()
    with output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, (name, original) in enumerate(rows, 1):
            wav = ds / 'wavs' / name
            review = reviews.get(name, {})
            flags = (review.get('issues') or '').strip()
            asr2 = ''
            error = ''
            try:
                if not wav.is_file():
                    raise FileNotFoundError(str(wav))
                segments, _ = model.transcribe(str(wav), language='vi', beam_size=5,
                                               vad_filter=False, condition_on_previous_text=False)
                asr2 = ' '.join(seg.text.strip() for seg in segments).strip()
                category, similarity, reasons = classify(original, asr2, flags, args.threshold)
            except Exception as exc:
                category, similarity, reasons = 'error', 0.0, 'ASR_failed'
                error = f'{type(exc).__name__}: {exc}'[:400]
            counts[category] += 1
            writer.writerow({
                'clip': name, 'is_fixed_test': 'yes' if name in test_ids else 'no',
                'review_status': review.get('status', 'unreviewed'), 'audit_flags': flags,
                'original_text': original, 'asr2_text': asr2,
                'similarity': f'{similarity:.4f}', 'category': category, 'reasons': reasons,
                'suggested_text': asr2 if category == 'needs_review' else original,
                'error': error,
            })
            f.flush()
            if idx % 10 == 0 or idx == len(rows):
                print(f'[{idx}/{len(rows)}] low_disagreement={counts["low_disagreement"]}, '
                      f'needs_review={counts["needs_review"]}, error={counts["error"]}', flush=True)
    summary = {'total': len(rows), 'counts': counts, 'seconds': round(time.time()-start,1),
               'scope': args.scope, 'model': args.model, 'threshold': args.threshold,
               'output': str(output), 'note': 'ASR agreement is not verified ground truth.'}
    report = output.with_suffix('.summary.json')
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nDONE\n' + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
