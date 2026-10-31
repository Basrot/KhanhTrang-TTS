#!/usr/bin/env python3
"""Create a reproducible, non-destructive, source-grouped dataset split.

Requires the existing metadata, source manifest, fixed test.csv, ASR second-opinion
CSV and quality report. Never modifies the original files or copies audio.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_pipe(path):
    rows = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if '|' in line:
            name, text = line.split('|', 1)
            rows.append((Path(name).name, text.strip()))
    return rows


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=r'C:\Khanhtrang-tts')
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', help='New, nonexistent output directory')
    args = ap.parse_args()
    if not 0.05 <= args.val_ratio <= 0.40:
        ap.error('--val-ratio must be 0.05..0.40')
    root = Path(args.root)
    paths = {
        'new': root/'data/f5tts_dataset/metadata.csv',
        'old': root/'data/f5tts_dataset/metadata_ten_cu.csv',
        'manifest': root/'data/processed/segments_manifest.csv',
        'test': root/'eval/data/test.csv',
        'asr': root/'eval/dataset_audit/asr_second_opinion_development.csv',
        'quality': root/'eval/dataset_audit/train_val_quality_report.csv',
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f'Missing {label}: {path}')
    new, old = read_pipe(paths['new']), read_pipe(paths['old'])
    manifest = read_csv(paths['manifest'])
    if not len(new) == len(old) == len(manifest):
        raise SystemExit('Metadata and manifest row counts differ; refusing positional mapping.')
    if len({n for n, _ in new}) != len(new):
        raise SystemExit('Duplicate clip filenames in new metadata')
    seg2src = {r['segment_file']: r['source_file'] for r in manifest}
    if len(seg2src) != len(manifest):
        raise SystemExit('Duplicate segment filenames in manifest')
    mapping = {}
    original = {}
    for (name, text), (old_name, _) in zip(new, old):
        if old_name not in seg2src:
            raise SystemExit(f'Missing old segment mapping: {old_name}')
        mapping[name] = seg2src[old_name]
        original[name] = text
    # This check cannot establish positional alignment if the old/new files
    # have been reordered; it only checks mapping completeness.
    test_rows = read_csv(paths['test'])
    test_names = {Path(r['wav']).name for r in test_rows}
    if len(test_names) != len(test_rows) or not test_names <= mapping.keys():
        raise SystemExit('Test has duplicates or unmapped clips')
    test_sources = {mapping[n] for n in test_names}
    asr_rows = read_csv(paths['asr'])
    quality_rows = read_csv(paths['quality'])
    quality = {r['clip']: r for r in quality_rows}
    if len(quality) != len(quality_rows):
        raise SystemExit('Duplicate quality clip IDs')
    if len({r['clip'] for r in asr_rows}) != len(asr_rows):
        raise SystemExit('Duplicate ASR clip IDs')
    expected_dev = {n for n in mapping if mapping[n] not in test_sources}
    if {r['clip'] for r in asr_rows} != expected_dev:
        raise SystemExit('ASR CSV does not cover exactly all development clips')
    if set(quality) != expected_dev:
        raise SystemExit('Quality report does not cover exactly all development clips')
    fields = ['clip', 'wav', 'source_video', 'text', 'asr2_text', 'similarity', 'review_status', 'audit_flags', 'category', 'reasons']
    selected, quarantine = [], []
    for r in asr_rows:
        n = r['clip']
        q = quality[n]
        if q['source_video'] != mapping[n]:
            raise SystemExit(f'Source mismatch for {n}')
        if q['original_text'] != original[n] or r['original_text'] != original[n]:
            raise SystemExit(f'Transcript mismatch for {n}; files changed after ASR run')
        wav = root/'data/f5tts_dataset/wavs'/n
        if not wav.is_file():
            raise SystemExit(f'Missing audio: {wav}')
        row = {
            'clip': n, 'wav': str(wav), 'source_video': mapping[n],
            'text': original[n], 'asr2_text': r.get('asr2_text',''),
            'similarity': r.get('similarity',''),
            'review_status': r.get('review_status', q.get('status','')),
            'audit_flags': r.get('audit_flags', q.get('issues','')),
            'category': r.get('category',''), 'reasons': r.get('reasons','')
        }
        if row['category'] == 'low_disagreement' and not row['audit_flags'].strip() and not r.get('error','').strip():
            selected.append(row)
        else:
            quarantine.append(row)
    groups = {}
    for row in selected:
        groups.setdefault(row['source_video'], []).append(row)
    if len(groups) < 3:
        raise SystemExit('Not enough source groups to split train and validation')
    names = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(names)
    target = len(selected) * args.val_ratio
    # Pick group subset close to target, never using all groups.
    # Dynamic programming over possible clip counts (up to 791).
    dp = {0: ()}
    for g in names:
        count = len(groups[g])
        for s, chosen in list(dp.items()):
            ns = s + count
            if ns not in dp:
                dp[ns] = chosen + (g,)
    candidates = [(abs(s-target), abs(len(chosen)-round(len(groups)*args.val_ratio)), s, chosen)
                  for s, chosen in dp.items() if 0 < len(chosen) < len(groups)]
    _, _, _, chosen = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    val_sources = set(chosen)
    train = sorted((r for r in selected if r['source_video'] not in val_sources), key=lambda r:r['clip'])
    val = sorted((r for r in selected if r['source_video'] in val_sources), key=lambda r:r['clip'])
    assert not ({r['source_video'] for r in train} & {r['source_video'] for r in val})
    assert not (({r['source_video'] for r in train} | {r['source_video'] for r in val}) & test_sources)
    assert not ({r['clip'] for r in train} & {r['clip'] for r in val})
    out = Path(args.output) if args.output else root/'eval'/('dataset_v2_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if out.exists():
        raise SystemExit(f'Output already exists, refusing overwrite: {out}')
    out.mkdir(parents=True)
    write_csv(out/'train.csv', train, fields)
    write_csv(out/'validation.csv', val, fields)
    write_csv(out/'quarantine.csv', sorted(quarantine, key=lambda r:r['clip']), fields)
    write_csv(out/'test_frozen.csv', test_rows, list(test_rows[0].keys()))
    excluded = [{'clip': n, 'source_video': mapping[n], 'is_fixed_test': 'yes' if n in test_names else 'no'}
                for n in sorted(mapping) if mapping[n] in test_sources]
    write_csv(out/'excluded_test_sources.csv', excluded, ['clip','source_video','is_fixed_test'])
    report = {
        'seed': args.seed, 'requested_validation_ratio': args.val_ratio,
        'train_clips': len(train), 'validation_clips': len(val),
        'quarantine_clips': len(quarantine), 'fixed_test_clips': len(test_rows),
        'excluded_test_source_clips': len(excluded),
        'train_source_count': len({r['source_video'] for r in train}),
        'validation_source_count': len(val_sources),
        'test_source_count': len(test_sources),
        'actual_validation_ratio': round(len(val)/len(selected),4),
        'validation_sources': sorted(val_sources),
        'source_files_sha256': {k:digest(p) for k,p in paths.items()},
        'limitations': [
            'ASR agreement is not verified ground truth.',
            'Source mapping relies on positional alignment of old/new metadata; independently verify before training.',
            'Source names alone do not establish single-speaker purity or absence of duplicate audio.',
            'This script does not train a model or change existing datasets.'
        ]
    }
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('=== DATASET V2 CREATED ===')
    for k in ('train_clips','validation_clips','quarantine_clips','fixed_test_clips','excluded_test_source_clips','train_source_count','validation_source_count','test_source_count','actual_validation_ratio'):
        print(f'{k}: {report[k]}')
    print(f'Output: {out}')

if __name__ == '__main__':
    main()
