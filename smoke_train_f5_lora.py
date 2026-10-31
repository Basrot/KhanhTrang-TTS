import sys
import gc
from pathlib import Path

import torch
import torchaudio
from datasets import load_from_disk
from torch.utils.checkpoint import checkpoint as activation_checkpoint

sys.path.insert(0, r"C:\Khanhtrang-tts\F5-TTS-Vietnamese\src")

from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer

ROOT = Path(r"C:\Khanhtrang-tts")
REPO = ROOT / "F5-TTS-Vietnamese"
DATA = ROOT / "eval" / "dataset_v2_20261029_104818" / "f5_export"
CHECKPOINT = REPO / "model_last.pt"

torch.manual_seed(42)

class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.lora_A = torch.nn.Linear(
            base.in_features, rank, bias=False
        )
        self.lora_B = torch.nn.Linear(
            rank, base.out_features, bias=False
        )
        self.scale = alpha / rank

        torch.nn.init.kaiming_uniform_(
            self.lora_A.weight, a=5**0.5
        )
        torch.nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_B(
            self.lora_A(x)
        ) * self.scale


print("=== 1. TOKENIZER ===", flush=True)

vocab_path = REPO / "data" / "Emilia_ZH_EN" / "vocab.txt"

# Use the repository's main vocab instead of Emilia if present.
vocab_path = REPO / "vocab.txt"

if not vocab_path.exists():
    raise FileNotFoundError(vocab_path)

vocab = vocab_path.read_text(encoding="utf-8").splitlines()
vocab_map = {
    char: index for index, char in enumerate(vocab)
}

print("Vocab size:", len(vocab_map), flush=True)


print("=== 2. LOAD CHECKPOINT ===", flush=True)

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

state = {
    k.removeprefix("transformer."): v
    for k, v in ckpt["model_state_dict"].items()
    if k.startswith("transformer.")
}


print("=== 3. CREATE MODEL ===", flush=True)

dit = DiT(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    text_num_embeds=2566,
    conv_layers=4,
    mel_dim=100,
    text_mask_padding=True,
    pe_attn_head=None,
)

dit.load_state_dict(state, strict=True)

del ckpt, state
gc.collect()

for p in dit.parameters():
    p.requires_grad = False

for block in dit.transformer_blocks:
    for name in ["to_q", "to_v"]:
        setattr(
            block.attn,
            name,
            LoRALinear(getattr(block.attn, name))
        )

# Activation checkpointing for transformer blocks.
for block in dit.transformer_blocks:
    original_forward = block.forward

    def make_checkpointed_forward(fn):
        def wrapped(*args, **kwargs):
            if not torch.is_grad_enabled():
                return fn(*args, **kwargs)

            return activation_checkpoint(
                fn,
                *args,
                use_reentrant=False,
                **kwargs
            )

        return wrapped

    block.forward = make_checkpointed_forward(original_forward)


print("=== 4. LOAD TRAIN SAMPLE ===", flush=True)

ds = load_from_disk(str(DATA / "train" / "raw"))

rows = sorted(
    range(len(ds)),
    key=lambda i: ds[i]["duration"]
)

row = ds[rows[0]]

print("Duration:", row["duration"], flush=True)
print("Text:", row["text"][:120], flush=True)

audio, sr = torchaudio.load(row["audio_path"])

if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=True)

if sr != 24000:
    audio = torchaudio.functional.resample(
        audio, sr, 24000
    )

# Limit the smoke test to 3 seconds.
audio = audio[:, :24000 * 3]

if audio.shape[-1] < 24000:
    raise RuntimeError("Audio sample too short")

print("Audio shape:", tuple(audio.shape), flush=True)


print("=== 5. GPU ===", flush=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = torch.device("cuda")

model = CFM(
    transformer=dit,
    vocab_char_map=vocab_map,
    mel_spec_kwargs={
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 100,
        "target_sample_rate": 24000,
        "mel_spec_type": "vocos",
    },
)

model = model.to(device=device, dtype=torch.float16)

audio = audio.to(device)

torch.cuda.reset_peak_memory_stats()

print(
    "Allocated before forward:",
    round(torch.cuda.memory_allocated() / 1024**3, 3),
    "GB",
    flush=True
)


print("=== 6. FORWARD ===", flush=True)

with torch.autocast("cuda", dtype=torch.float16):
    loss, _, _ = model(
        inp=audio,
        text=[row["text"]]
    )

print("Loss:", float(loss.detach()), flush=True)

if not torch.isfinite(loss):
    raise RuntimeError("Loss is NaN or Inf")


print("=== 7. BACKWARD ===", flush=True)

# Scaling reduces the risk of FP16 gradient underflow.
scale = 1024.0
(loss * scale).backward()

trainable = [
    p for p in model.parameters()
    if p.requires_grad
]

grad_count = sum(
    p.grad is not None for p in trainable
)

print("Parameters with gradients:", grad_count, flush=True)

if grad_count == 0:
    raise RuntimeError("No LoRA gradients")

for p in trainable:
    if p.grad is not None:
        p.grad.div_(scale)
        if not torch.isfinite(p.grad).all():
            raise RuntimeError("NaN or Inf gradient")

torch.cuda.synchronize()

print(
    "Peak allocated VRAM:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    "GB",
    flush=True
)

print("PASS: forward/backward completed", flush=True)
