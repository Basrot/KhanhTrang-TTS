#!/usr/bin/env python3
"""Export frozen dataset_v2 CSV manifests into F5-TTS CustomDatasetPath layout.
Requires: pip install datasets soundfile pyarrow
Does not change source CSVs/WAVs or existing F5 data.
"""
import argparse
import csv
import json
import math
import os
import sys
import wave
from pathlib import Path


def duration_seconds(path):
    try:
        import soundfile as sf
        info = sf.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError('invalid audio info')
        return info.frames / info.samplerate
    except ImportError:
        with wave.open(str(path), 'rb') as f:
            return f.getnframes() / f.getframerate()


def load_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'clip', 'wav', 'source_video', 'text'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f'{path}: missing columns {sorted(required - set(reader.fieldnames or []))}')
        return list(reader)


def load_test_sources(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'clip', 'source_video', 'is_fixed_test'}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f'{path}: missing required columns')
        rows = list(reader)
    mapping = {r['clip']: r['source_video'] for r in rows}
    if len(mapping) != len(rows):
        raise SystemExit('Duplicate clip in excluded_test_sources.csv')
    frozen = {r['clip'] for r in rows if r['is_fixed_test'].strip().lower() == 'yes'}
    if len(frozen) != 30:
        raise SystemExit(f'Expected 30 marked fixed test clips, got {len(frozen)}')
    return mapping


def read_fixed_markers(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return [r for r in csv.DictReader(f) if r['is_fixed_test'].strip().lower() == 'yes']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(r'C:\Khanhtrang-tts'))
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--min-seconds', type=float, default=0.3)
    p.add_argument('--max-seconds', type=float, default=30.0)
    args = p.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output = (args.output_dir or dataset_dir / 'f5_export').resolve()
    if output.exists():
        raise SystemExit(f'Output already exists: {output}. Specify a new --output-dir to avoid overwriting.')

    train = load_rows(dataset_dir / 'train.csv')
    val = load_rows(dataset_dir / 'validation.csv')
    # Fixed test is intentionally preserved in its original schema (often only 'wav').
    # Resolve its source groups using the frozen excluded-test-sources manifest.
    with (dataset_dir / 'test_frozen.csv').open('r', encoding='utf-8-sig', newline='') as f:
        test_reader = csv.DictReader(f)
        if 'wav' not in (test_reader.fieldnames or []):
            raise SystemExit('test_frozen.csv must have a wav column')
        test_original = list(test_reader)
    excluded_test = load_test_sources(dataset_dir / 'excluded_test_sources.csv')
    test = []
    for r in test_original:
        clip = Path(r['wav']).name
        if not clip or clip not in excluded_test:
            raise SystemExit(f'Fixed test clip not in excluded_test_sources.csv: {clip}')
        test.append({'clip': clip, 'source_video': excluded_test[clip], 'wav': r['wav']})
    if not train or not val or not test:
        raise SystemExit('Train, validation and fixed test must be non-empty.')
    if len(test) != 30:
        raise SystemExit(f'Expected 30 frozen test clips, found {len(test)}')
    if {r['clip'] for r in test} != {r['clip'] for r in read_fixed_markers(dataset_dir / 'excluded_test_sources.csv')}:
        raise SystemExit('Fixed test membership mismatch with excluded_test_sources.csv')

    all_groups = {'train': train, 'validation': val, 'test': test}
    ids = {k: {r['clip'] for r in rows} for k, rows in all_groups.items()}
    sources = {k: {r['source_video'] for r in rows} for k, rows in all_groups.items()}
    for k, rows in all_groups.items():
        if len(ids[k]) != len(rows):
            raise SystemExit(f'Duplicate clip in {k}')
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if ids[a] & ids[b] or sources[a] & sources[b]:
            raise SystemExit(f'LEAKAGE detected between {a} and {b}')

    prepared = {}
    stats = {}
    for group in ('train', 'validation'):
        valid = []
        excluded = []
        for r in all_groups[group]:
            path = Path(r['wav'])
            if not path.is_absolute():
                path = args.root / path
            path = path.resolve()
            if not path.is_file():
                raise SystemExit(f'Missing WAV: {path}')
            if path.name.lower() != r['clip'].lower():
                raise SystemExit(f'Clip/path mismatch: {r["clip"]} / {path}')
            transcript = r['text'].strip()
            if not transcript:
                raise SystemExit(f'Empty transcript: {r["clip"]}')
            dur = duration_seconds(path)
            if not math.isfinite(dur) or dur <= 0:
                raise SystemExit(f'Invalid WAV duration: {path}')
            row = {'audio_path': str(path), 'text': transcript, 'duration': round(dur, 6)}
            if args.min_seconds <= dur <= args.max_seconds:
                valid.append(row)
            else:
                excluded.append({'clip': r['clip'], 'duration': round(dur, 6), 'reason': 'duration_out_of_range'})
        if not valid:
            raise SystemExit(f'No valid rows in {group}')
        prepared[group] = (valid, excluded)
        stats[group] = {'input': len(all_groups[group]), 'exported': len(valid), 'excluded': len(excluded), 'hours': round(sum(r['duration'] for r in valid)/3600, 4)}

    try:
        from datasets import Dataset
    except ImportError as e:
        raise SystemExit('Missing datasets package. Install with: .\\venv\\Scripts\\python.exe -m pip install datasets soundfile pyarrow') from e

    # Create output only after all validation passes.
    output.mkdir(parents=True)
    for group, (rows, excluded) in prepared.items():
        dest = output / group
        dest.mkdir()
        Dataset.from_list(rows).save_to_disk(str(dest / 'raw'))
        with (dest / 'duration.json').open('w', encoding='utf-8') as f:
            json.dump({'duration': [r['duration'] for r in rows]}, f, ensure_ascii=False, indent=2)
        with (dest / 'excluded.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['clip', 'duration', 'reason'])
            writer.writeheader()
            writer.writerows(excluded)
        # Confirm dataset is loadable and aligned.
        from datasets import load_from_disk
        loaded = load_from_disk(str(dest / 'raw'))
        assert len(loaded) == len(rows)
        assert all(abs(loaded[i]['duration'] - rows[i]['duration']) < 1e-5 for i in range(len(rows)))

    with (output / 'report.json').open('w', encoding='utf-8') as f:
        json.dump({'source': str(dataset_dir), 'output': str(output), 'stats': stats,
                   'note': 'ASR agreement is not verified ground truth; speaker identity is not verified.'},
                  f, ensure_ascii=False, indent=2)
    print('=== F5 DATASET EXPORTED ===')
    for k, v in stats.items():
        print(f'{k}: {v["exported"]}/{v["input"]} clips, excluded={v["excluded"]}, hours={v["hours"]}')
    print('Output:', output)
    print('Use CustomDatasetPath with the train or validation subdirectory.')

if __name__ == '__main__':
    main()
