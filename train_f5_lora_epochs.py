"""F5-TTS LoRA epoch trainer. Requires train_f5_lora.py alongside this file.
Run: python train_f5_lora_epochs.py --epochs 10 --accum 8 --max-seconds 30
Resume: python train_f5_lora_epochs.py --epochs 10 --accum 8 --resume
"""
import argparse
import gc
import json
import math
import random
from pathlib import Path

import torch
from datasets import load_from_disk

from train_f5_lora import ROOT, DATA, CHECKPOINT, create_model, load_audio, mel_from_audio


def adapter_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.transformer.state_dict().items()
            if '.lora_A.' in k or '.lora_B.' in k}


def restore_adapter(model, state):
    missing, unexpected = model.transformer.load_state_dict(state, strict=False)
    if unexpected or any('.lora_A.' in k or '.lora_B.' in k for k in missing):
        raise RuntimeError(f'Adapter state mismatch: unexpected={unexpected}; missing LoRA={missing}')


def checkpoint_save(path, model, optimizer, scaler, epoch, best_val, args):
    payload = dict(epoch=epoch, adapter=adapter_state(model), optimizer=optimizer.state_dict(),
                   scaler=scaler.state_dict(), best_val=best_val,
                   config=dict(accum=args.accum, lr=args.lr, max_seconds=args.max_seconds,
                               rank=8, alpha=16, seed=args.seed),
                   base_checkpoint=str(CHECKPOINT),
                   torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all())
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def selected_rows(ds, max_seconds):
    return [i for i in range(len(ds)) if 1 <= float(ds[i]['duration']) <= max_seconds]


def loss_for_row(model, row, max_seconds):
    wav = load_audio(row, max_seconds)
    mel = mel_from_audio(model, wav)
    with torch.autocast('cuda', dtype=torch.float16):
        loss, _, _ = model(inp=mel, text=[row['text']])
    return loss


def validate(model, ds, indices, max_seconds, seed):
    # Preserve RNG and fix diffusion noise/masks for comparable validation epochs.
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    py_rng = random.getstate()
    random.seed(seed + 777)
    torch.manual_seed(seed + 777)
    torch.cuda.manual_seed_all(seed + 777)
    model.eval()
    total = 0.0
    try:
        with torch.no_grad():
            for i in indices:
                loss = loss_for_row(model, ds[i], max_seconds)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite validation loss at row {i}')
                total += float(loss.detach())
                del loss
    finally:
        torch.set_rng_state(torch_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(py_rng)
    return total / len(indices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10, help='Total desired epochs, including resumed epochs')
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--max-seconds', type=float, default=30)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'f5_lora_epochs_v1')
    args = p.parse_args()
    if args.epochs < 1 or args.accum < 1 or not (1 <= args.max_seconds <= 30):
        p.error('epochs and accum must be >=1; max-seconds must be in [1,30]')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / 'checkpoint_latest.pt'
    best = args.output / 'checkpoint_best.pt'
    if latest.exists() and not args.resume:
        raise RuntimeError(f'Checkpoint already exists: {latest}. Use --resume or a different --output.')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train = load_from_disk(str(DATA / 'train' / 'raw'))
    val = load_from_disk(str(DATA / 'validation' / 'raw'))
    train_ids = selected_rows(train, args.max_seconds)
    val_ids = selected_rows(val, args.max_seconds)
    if not train_ids or not val_ids:
        raise RuntimeError('No eligible train/validation clips; inspect max-seconds')
    print(f'Train {len(train_ids)}/{len(train)} | validation {len(val_ids)}/{len(val)} | max {args.max_seconds}s', flush=True)
    if len(train_ids) != len(train) or len(val_ids) != len(val):
        print('WARNING: some clips are excluded by max-seconds. Use 30 for full dataset.', flush=True)
    model = create_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print('Trainable parameters:', sum(p.numel() for p in trainable), flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler('cuda') if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()
    start_epoch, best_val = 1, math.inf
    if args.resume:
        if not latest.exists():
            raise FileNotFoundError(latest)
        state = torch.load(latest, map_location='cpu', weights_only=False)
        old = state['config']
        for key in ('accum', 'lr', 'max_seconds', 'seed'):
            if old[key] != getattr(args, key):
                raise RuntimeError(f'Cannot resume with changed {key}: saved {old[key]}, current {getattr(args,key)}')
        restore_adapter(model, state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_epoch = state['epoch'] + 1
        best_val = state['best_val']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        print('Resumed after epoch', state['epoch'], flush=True)
        del state
        gc.collect()
    torch.cuda.reset_peak_memory_stats()
    with (args.output / 'history.jsonl').open('a', encoding='utf-8') as log:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            order = train_ids[:]
            random.shuffle(order)
            running = 0.0
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(order), args.accum):
                batch = order[start:start + args.accum]
                # Normalize by actual accumulation count, including last partial group.
                for i in batch:
                    loss = loss_for_row(model, train[i], args.max_seconds)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite train loss: epoch={epoch}, row={i}')
                    running += float(loss.detach())
                    scaler.scale(loss / len(batch)).backward()
                    del loss
                scaler.unscale_(optimizer)
                grads = [p.grad for p in trainable if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                    raise RuntimeError(f'Invalid LoRA gradients: epoch={epoch}, group={start}')
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < old_scale:
                    raise RuntimeError(f'GradScaler skipped update: epoch={epoch}, group={start}')
                optimizer.zero_grad(set_to_none=True)
            train_loss = running / len(order)
            val_loss = validate(model, val, val_ids, args.max_seconds, args.seed)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            record = dict(epoch=epoch, train_loss=round(train_loss, 6), val_loss=round(val_loss, 6),
                          train_clips=len(order), val_clips=len(val_ids),
                          optimizer_steps=math.ceil(len(order) / args.accum), peak_allocated_gb=round(peak, 3))
            print(json.dumps(record, ensure_ascii=False), flush=True)
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
            log.flush()
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            checkpoint_save(latest, model, optimizer, scaler, epoch, best_val, args)
            if improved:
                checkpoint_save(best, model, optimizer, scaler, epoch, best_val, args)
    print('DONE:', latest, flush=True)

if __name__ == '__main__':
    main()
