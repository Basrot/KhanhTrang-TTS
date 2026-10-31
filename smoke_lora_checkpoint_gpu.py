import sys
import gc
import torch

sys.path.insert(0, r"C:\Khanhtrang-tts\F5-TTS-Vietnamese\src")

from f5_tts.model import DiT

class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.lora_A = torch.nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank

        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale

checkpoint_path = r"C:\Khanhtrang-tts\F5-TTS-Vietnamese\model_last.pt"

print("Loading checkpoint on CPU...")
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

state = {
    k.removeprefix("transformer."): v
    for k, v in checkpoint["model_state_dict"].items()
    if k.startswith("transformer.")
}

print("Creating DiT...")
model = DiT(
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

model.load_state_dict(state, strict=True)
print("PASS: checkpoint loaded")

del checkpoint, state
gc.collect()

for p in model.parameters():
    p.requires_grad = False

for block in model.transformer_blocks:
    for name in ["to_q", "to_v"]:
        setattr(block.attn, name, LoRALinear(getattr(block.attn, name)))

print("PASS: LoRA attached")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model = model.to(device="cuda", dtype=torch.float16)

torch.cuda.synchronize()

print("GPU:", torch.cuda.get_device_name(0))
print("Allocated VRAM:", round(torch.cuda.memory_allocated() / 1024**3, 3), "GB")
print("Peak VRAM:", round(torch.cuda.max_memory_allocated() / 1024**3, 3), "GB")
print("Trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("PASS: model on GPU")
