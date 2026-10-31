"""F5-TTS LoRA training for a local Windows dataset; independent of upstream Trainer.
Usage: python train_f5_lora.py --steps 10
"""
import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import torch
import torchaudio
from datasets import load_from_disk
from torch.utils.checkpoint import checkpoint as activation_checkpoint

ROOT = Path(r'C:\Khanhtrang-tts')
REPO = ROOT / 'F5-TTS-Vietnamese'
DATA = ROOT / 'eval' / 'dataset_v2_20261029_104818' / 'f5_export'
CHECKPOINT = REPO / 'model_last.pt'
OUTPUT = ROOT / 'outputs' / 'f5_lora_v2'
sys.path.insert(0, str(REPO / 'src'))
from f5_tts.model import CFM, DiT

class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.lora_A = torch.nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** .5)
        torch.nn.init.zeros_(self.lora_B.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)
    def forward(self, x):
        # Base and adapter use the same compute dtype; adapter parameters remain FP32.
        base_out = self.base(x)
        adapter = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype)))
        return base_out + adapter.to(base_out.dtype) * self.scale

def create_model():
    print('Loading original checkpoint on CPU...', flush=True)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    state = {k.removeprefix('transformer.'): v for k, v in ckpt['model_state_dict'].items() if k.startswith('transformer.')}
    dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
              text_num_embeds=2566, conv_layers=4, mel_dim=100,
              text_mask_padding=False, pe_attn_head=1)
    dit.load_state_dict(state, strict=True)
    del ckpt, state
    gc.collect()
    for p in dit.parameters():
        p.requires_grad_(False)
    # Cast frozen backbone only, then attach FP32 adapters.
    dit = dit.to(device='cuda', dtype=torch.float16)
    for block in dit.transformer_blocks:
        for name in ('to_q', 'to_v'):
            setattr(block.attn, name, LoRALinear(getattr(block.attn, name)).to('cuda'))
    for block in dit.transformer_blocks:
        forward = block.forward
        def wrap(fn):
            def call(*args, **kwargs):
                if not torch.is_grad_enabled():
                    return fn(*args, **kwargs)
                return activation_checkpoint(fn, *args, use_reentrant=False, **kwargs)
            return call
        block.forward = wrap(forward)
    vocab = (REPO / 'vocab.txt').read_text(encoding='utf-8').splitlines()
    if len(vocab) != 2566 or len(set(vocab)) != len(vocab):
        raise RuntimeError('Unexpected vocabulary. Verify checkpoint/tokenizer mapping.')
    vocab_map = {ch: i for i, ch in enumerate(vocab)}
    model = CFM(transformer=dit, vocab_char_map=vocab_map,
                mel_spec_kwargs={'n_fft':1024, 'hop_length':256, 'win_length':1024,
                                 'n_mel_channels':100, 'target_sample_rate':24000,
                                 'mel_spec_type':'vocos'})
    # Keep mel computation in FP32 on CPU, and frozen DiT in FP16.
    model.mel_spec = model.mel_spec.to('cpu', dtype=torch.float32)
    return model

def load_audio(row, seconds):
    wav, sr = torchaudio.load(row['audio_path'])
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    if seconds:
        wav = wav[:, :int(seconds * 24000)]
    if wav.shape[-1] < 24000:
        raise RuntimeError('Audio under 1 second')
    return wav

def mel_from_audio(model, wav):
    with torch.no_grad():
        mel = model.mel_spec(wav).permute(0, 2, 1).contiguous()
    return mel.to('cuda', dtype=torch.float16)

def save_adapter(model, optimizer, step, output):
    adapter = {k: v.detach().cpu().clone() for k,v in model.transformer.state_dict().items()
               if '.lora_A.' in k or '.lora_B.' in k}
    tmp = output / 'adapter_latest.tmp'
    torch.save({'step':step, 'adapter':adapter,
                'optimizer':optimizer.state_dict(), 'architecture':'F5TTS_v1_Base',
                'rank':8, 'alpha':16, 'targets':['to_q','to_v'],
                'base_checkpoint':str(CHECKPOINT)}, tmp)
    tmp.replace(output / 'adapter_latest.pt')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--max-seconds', type=float, default=3.0,
                    help='Smoke-test cap; increase for full-clip training')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save-every', type=int, default=10)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(str(DATA / 'train' / 'raw'))
    valid = [i for i in range(len(ds)) if 1 <= ds[i]['duration'] <= 30]
    if not valid:
        raise RuntimeError('No valid train clips')
    model = create_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print('Trainable parameters:', sum(p.numel() for p in trainable), flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    # We deliberately do not resume the original optimizer/update=540000.
    scaler = torch.amp.GradScaler('cuda', enabled=True) if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    log_path = OUTPUT / 'smoke_log.jsonl'
    with log_path.open('a', encoding='utf-8') as log:
        for step in range(1, args.steps + 1):
            row = ds[random.choice(valid)]
            wav = load_audio(row, args.max_seconds)
            mel = mel_from_audio(model, wav)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss, _, _ = model(inp=mel, text=[row['text']])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}: {loss.item()}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = [p.grad for p in trainable if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError(f'Invalid/missing LoRA gradients at step {step}')
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < prev_scale:
                raise RuntimeError(f'Optimizer step skipped by GradScaler at step {step}')
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            item = {'step':step, 'loss':round(loss.item(), 6),
                    'duration_used':round(wav.shape[-1]/24000, 3),
                    'peak_allocated_gb':round(peak,3)}
            print(json.dumps(item), flush=True)
            log.write(json.dumps(item, ensure_ascii=False)+'\n')
            log.flush()
            del wav, mel, loss
            if step % args.save_every == 0 or step == args.steps:
                save_adapter(model, optimizer, step, OUTPUT)
    print('PASS: training steps completed. Adapter:', OUTPUT / 'adapter_latest.pt', flush=True)

if __name__ == '__main__':
    main()
