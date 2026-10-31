import sys
import torch

sys.path.insert(0, r"C:\Khanhtrang-tts\F5-TTS-Vietnamese\src")

from f5_tts.model import DiT

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

for p in model.parameters():
    p.requires_grad = False

count = 0

for block in model.transformer_blocks:
    for name in ["to_q", "to_v"]:
        setattr(
            block.attn,
            name,
            LoRALinear(getattr(block.attn, name))
        )
        count += 1

total = sum(p.numel() for p in model.parameters())
trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("LoRA modules:", count)
print("Total parameters:", total)
print("Trainable parameters:", trainable)
print("Trainable %:", round(trainable / total * 100, 4))
print("PASS: LoRA modules attached")
