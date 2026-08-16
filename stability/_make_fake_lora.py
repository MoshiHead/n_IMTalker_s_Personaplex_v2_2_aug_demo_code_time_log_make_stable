"""Write a real .safetensors file shaped like the reference LoRA checkpoint,
using only the standard library, so derive_ref_lora_config.py can be tested
end to end without the real 600MB weights.
"""
import json
import pathlib
import struct
import sys

out_dir = pathlib.Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)

R = 128
DIM = 4096

# The module set the log implies: 32 trained modules across transformer and
# depformer, mixing several suffixes.
modules = []
for i in range(8):
    modules.append(f"transformer.layers.{i}.self_attn.in_proj")
    modules.append(f"transformer.layers.{i}.self_attn.out_proj")
for i in range(6):
    modules.append(f"depformer.layers.{i}.self_attn.in_proj")
for i in range(10):
    modules.append(f"transformer.layers.{i}.gating.linear_in")

header = {}
offset = 0
blobs = []
for name in modules:
    for ab, shape in (("A", [R, DIM]), ("B", [DIM, R])):
        key = f"base_model.model.{name}.lora_{ab}.default.weight"
        nbytes = shape[0] * shape[1] * 2  # bf16
        header[key] = {"dtype": "BF16", "shape": shape,
                       "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
        blobs.append(b"\x00" * nbytes)

# A non-LoRA key that must be ignored rather than crashing the parser.
header["base_model.model.transformer.norm.weight"] = {
    "dtype": "BF16", "shape": [DIM], "data_offsets": [offset, offset + DIM * 2]}
blobs.append(b"\x00" * (DIM * 2))

header["__metadata__"] = {"format": "pt"}
header_bytes = json.dumps(header).encode("utf-8")

path = out_dir / "adapter_model.safetensors"
with path.open("wb") as f:
    f.write(struct.pack("<Q", len(header_bytes)))
    f.write(header_bytes)
    for b in blobs:
        f.write(b)

print(f"[ok] wrote {path} ({path.stat().st_size / 1e6:.1f} MB), "
      f"{len(modules)} trained modules, r={R}")

# Also write the WRONG hand-written config the download script ships, so the
# --check path has something realistic to reject.
bad_config = {
    "peft_type": "LORA", "task_type": "FEATURE_EXTRACTION",
    "r": 128, "lora_alpha": 256.0, "lora_dropout": 0.05,
    "target_modules": ["proj", "fc1", "out_proj", "fc2", "linear", "in_proj"],
    "inference_mode": True, "init_lora_weights": True,
}
(out_dir / "adapter_config.json").write_text(json.dumps(bad_config, indent=2))
print("[ok] wrote the suffix-style adapter_config.json that ships today")
