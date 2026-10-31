import torch
from pathlib import Path

root = Path(r"C:\Khanhtrang-tts")
base_path = root / "F5-TTS-Vietnamese" / "model_last.pt"
lora_path = root / "outputs" / "f5_lora_base_v2" / "checkpoint_latest.pt"
output_path = root / "outputs" / "f5_lora_base_v2" / "model_merged_base_epoch10.pt"

print("Loading LoRA checkpoint...")
lora_ckpt = torch.load(
    lora_path, map_location="cpu", weights_only=True
)

adapter = lora_ckpt["adapter"]
config = lora_ckpt["config"]

rank = int(config["rank"])
alpha = float(config["alpha"])
scale = alpha / rank

print("Epoch:", lora_ckpt["epoch"])
print("Rank:", rank)
print("Alpha:", alpha)

print("Loading base checkpoint...")
base_ckpt = torch.load(
    base_path, map_location="cpu", weights_only=True
)

state = base_ckpt["model_state_dict"]

merged_count = 0

for key_a, A in adapter.items():
    if not key_a.endswith(".lora_A.weight"):
        continue

    key_b = key_a.replace(".lora_A.weight", ".lora_B.weight")

    if key_b not in adapter:
        raise RuntimeError(f"Missing LoRA B: {key_b}")

    B = adapter[key_b]

    layer_name = key_a.removesuffix(".lora_A.weight")
    target = "transformer." + layer_name + ".weight"

    if target not in state:
        raise RuntimeError(f"Missing base weight: {target}")

    W = state[target]

    delta = (B.float() @ A.float()) * scale

    if delta.shape != W.shape:
        raise RuntimeError(f"Shape mismatch: {target}")

    state[target] = (
        W.float() + delta
    ).to(dtype=W.dtype)

    merged_count += 1

if merged_count != 44:
    raise RuntimeError(
        f"Expected 44 merged layers, got {merged_count}"
    )

print("Merged layers:", merged_count)

# Keep the original checkpoint structure.
base_ckpt["model_state_dict"] = state

# EMA weights are no longer consistent after merging.
# Remove them from this inference export.
base_ckpt.pop("ema_model_state_dict", None)

output_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(base_ckpt, output_path)

print("PASS: LoRA merged successfully")
print("Output:", output_path)
print("Size GB:", round(output_path.stat().st_size / 1024**3, 3))
