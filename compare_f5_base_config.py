"""Compare original F5-TTS and LoRA-merged epoch-10 using the same reference/text/seed.
Run in C:\\Khanhtrang-tts with venv_f5train. No training, no modification of checkpoints.
"""
import argparse
import gc
import inspect
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(r'C:\Khanhtrang-tts')
REPO = ROOT / 'F5-TTS-Vietnamese'
sys.path.insert(0, str(REPO / 'src'))
from f5_tts.model import DiT
from f5_tts.infer import utils_infer as u

BASE = REPO / 'model_last.pt'
AFTER = ROOT / 'outputs' / 'f5_lora_epochs_v1' / 'model_merged_epoch10.pt'
REF = ROOT / 'data' / 'f5tts_dataset' / 'wavs' / 'clip_128.wav'
OUT = ROOT / 'outputs' / 'f5_comparison_base_config'
DEFAULT_TEXT = ('Xin chào quý vị và các bạn. Đây là bản tin tổng hợp trong ngày. '
                'Sau đây là những thông tin đáng chú ý về kinh tế, công nghệ và đời sống.')


def compatible_call(fn, **kwargs):
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def load_reference_text(ref_path, supplied):
    if supplied:
        return supplied
    # The training transcript of clip_128 is preferable to guessing its content.
    try:
        from datasets import load_from_disk
        data_root = ROOT / 'eval' / 'dataset_v2_20261029_104818' / 'f5_export'
        for split in ('train', 'validation'):
            p = data_root / split / 'raw'
            if not p.exists():
                continue
            ds = load_from_disk(str(p))
            for row in ds:
                if Path(str(row['audio_path'])).name.lower() == ref_path.name.lower():
                    transcript = str(row['text']).strip()
                    if transcript:
                        print(f'Reference transcript found in {split}: {transcript}')
                        return transcript
    except Exception as exc:
        print(f'Could not look up reference transcript: {exc}')
    raise RuntimeError('Reference transcript not found. Supply --ref-text "exact words spoken in clip_128.wav"')


def run_one(name, path, cfg, ref_audio, ref_text, text, vocoder, seed, steps):
    print(f'\n=== {name} ===\nCheckpoint: {path}', flush=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = u.load_model(
        model_cls=DiT, model_cfg=cfg, ckpt_path=str(path),
        mel_spec_type='vocos', vocab_file=str(REPO / 'vocab.txt'),
        use_ema=False, device='cuda'
    )
    model.eval()
    # Seed is reset directly before inference for identical sampling conditions.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        result = compatible_call(
            u.infer_process, ref_audio=ref_audio, ref_text=ref_text,
            gen_text=text, model_obj=model, model=model, vocoder=vocoder,
            mel_spec_type='vocos', nfe_step=steps, speed=1.0,
            show_info=print, device='cuda',
        )
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError('Unexpected infer_process return value')
    audio, sr = result[:2]
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) == 0:
        raise RuntimeError('Generated audio is invalid')
    outfile = OUT / f'{name}.wav'
    sf.write(str(outfile), audio, int(sr))
    print(f'SAVED {outfile} | {len(audio)/int(sr):.2f}s | {int(sr)} Hz', flush=True)
    del model, result, audio
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref-audio', default=str(REF))
    p.add_argument('--ref-text', default='')
    p.add_argument('--text', default=DEFAULT_TEXT)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=32)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate venv_f5train with GPU PyTorch')
    for f in (BASE, AFTER, REPO / 'vocab.txt', Path(args.ref_audio)):
        if not f.is_file():
            raise FileNotFoundError(f)
    if not hasattr(u, 'infer_process') or not hasattr(u, 'load_vocoder'):
        raise RuntimeError('Repo utils_infer lacks infer_process or load_vocoder')
    ref_text = load_reference_text(Path(args.ref_audio), args.ref_text)
    OUT.mkdir(parents=True, exist_ok=True)
    # Same DiT configuration as the training script.
    cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2,
               text_dim=512, text_mask_padding=False, pe_attn_head=1,
               conv_layers=4)
    print('Loading vocoder once...', flush=True)
    vocoder = compatible_call(u.load_vocoder, vocoder_name='vocos', is_local=False, device='cuda')
    ref_audio = str(Path(args.ref_audio))
    if hasattr(u, 'preprocess_ref_audio_text'):
        ref_audio, ref_text = compatible_call(
            u.preprocess_ref_audio_text, ref_audio_orig=ref_audio,
            ref_text=ref_text, show_info=print, device='cuda'
        )
    for name, path in [('f5_before', BASE), ('f5_after_epoch10', AFTER)]:
        run_one(name, path, cfg, ref_audio, ref_text, args.text, vocoder, args.seed, args.steps)
    print('\nPASS: both WAV files generated in', OUT, flush=True)


if __name__ == '__main__':
    main()
