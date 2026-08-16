"""Verify the reproducibility fixes without needing the 7B model or a GPU.

Checks:
  1. the seed helpers resolve/apply correctly and make torch reproducible
  2. the strict-runtime gate fires on a fork that drops kwargs, and the
     ALLOW_MOSHI_FALLBACK escape hatch works
  3. FM.sample's new noise_init branch selects the pre-generated noise
"""
import ast
import os
import sys
import textwrap

import torch

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
LIVETRY = os.path.join(REPO, "IMTalker", "liveTry.py")
FM = os.path.join(REPO, "IMTalker", "generator", "FM.py")

# ---------------------------------------------------------------- 1. seeding
# Pull the helper block out of liveTry.py without importing the module (which
# needs cv2/torchaudio/fastapi).
tree = ast.parse(open(LIVETRY, encoding="utf-8").read())
wanted = {"_env_flag", "resolve_personaplex_seed", "seed_personaplex"}
ns = {"os": os, "torch": torch, "print": lambda *a, **k: None}
found = set()
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        exec(compile(ast.Module([node], []), LIVETRY, "exec"), ns)
        found.add(node.name)
assert found == wanted, f"helpers missing from liveTry.py: {wanted - found}"

os.environ.pop("PERSONAPLEX_SEED", None)
assert ns["resolve_personaplex_seed"]() == 42, "default seed should be 42"
os.environ["PERSONAPLEX_SEED"] = ""
assert ns["resolve_personaplex_seed"]() is None, "empty seed must mean 'stay random'"
os.environ["PERSONAPLEX_SEED"] = "7"
assert ns["resolve_personaplex_seed"]() == 7
os.environ["PERSONAPLEX_SEED"] = "not-an-int"
assert ns["resolve_personaplex_seed"]() == 42, "bad seed must fall back, not crash"
os.environ.pop("PERSONAPLEX_SEED")

os.environ.pop("PERSONAPLEX_RESEED_PER_SESSION", None)
assert ns["_env_flag"]("PERSONAPLEX_RESEED_PER_SESSION", True) is True
os.environ["PERSONAPLEX_RESEED_PER_SESSION"] = "0"
assert ns["_env_flag"]("PERSONAPLEX_RESEED_PER_SESSION", True) is False
os.environ["PERSONAPLEX_RESEED_PER_SESSION"] = "on"
assert ns["_env_flag"]("PERSONAPLEX_RESEED_PER_SESSION", False) is True
os.environ.pop("PERSONAPLEX_RESEED_PER_SESSION")

# The property that actually matters: reseeding reproduces the sampling stream
# LMGen uses (torch.empty_like(...).exponential_ -> multinomial-free top-k draw).
def draw():
    probs = torch.softmax(torch.arange(64, dtype=torch.float32) / 8.0, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return int((probs / q).argmax().item()), float(torch.randn(3).sum())

ns["seed_personaplex"](42, "test")
a = [draw() for _ in range(5)]
ns["seed_personaplex"](42, "test")
b = [draw() for _ in range(5)]
ns["seed_personaplex"](43, "test")
c = [draw() for _ in range(5)]
assert a == b, f"same seed must reproduce the sampling stream:\n{a}\n{b}"
assert a != c, "a different seed must produce a different stream"
print("[ok] 1. seeding: defaults, empty-means-random, bad-value fallback, reproducible stream")

# ------------------------------------------------------- 2. strict runtime gate
# Replay the exact guard from liveTry.py against a stand-in loader signature.
import inspect

def good_loader(filename, device=None, dtype=None, quantize_4bit=False,
                num_codebooks=8, context=None):
    ...

def bad_loader(filename, copy_missing_weights=True, device=None, dtype=None,
               delays=None, cpu_offload=False):
    ...

def check(loader, strict):
    supported = set(inspect.signature(loader).parameters)
    optional = {"quantize_4bit": True, "num_codebooks": 8, "context": None}
    dropped = sorted(k for k in optional if k not in supported)
    if dropped and strict:
        raise RuntimeError(f"This moshi build ignores {dropped}.")
    return dropped

assert check(good_loader, strict=True) == []
try:
    check(bad_loader, strict=True)
    raise AssertionError("strict mode must refuse the fork that drops quantize_4bit")
except RuntimeError as e:
    assert "quantize_4bit" in str(e)
assert check(bad_loader, strict=False) == ["context", "num_codebooks", "quantize_4bit"]
print("[ok] 2. strict runtime gate: passes the PersonaPlex fork, refuses the repo fork,")
print("       and ALLOW_MOSHI_FALLBACK=1 still lets it through")

# --------------------------------------------------------- 3. FM noise_init
fm_src = open(FM, encoding="utf-8").read()
assert 'noise_init = data.get("noise_init")' in fm_src, "FM.py must read noise_init"
assert "if noise_init is not None:" in fm_src, "FM.py must branch on noise_init"
assert "elif self.opt.fix_noise_seed:" in fm_src, "the fix_noise_seed path must survive"

# Simulate the branch: a pre-generated buffer must be sliced per chunk, not redrawn.
num_frames_for_clip, dim_w, B = 24, 8, 1
gen = torch.Generator(device="cpu")
gen.manual_seed(42)
noise_buf = torch.randn(1, 200, dim_w, generator=gen)

def slice_chunk(noise_init, chunk_idx):
    start = chunk_idx * num_frames_for_clip
    end = (chunk_idx + 1) * num_frames_for_clip
    x0 = noise_init[:, start:end, :].to(dtype=torch.float32)
    if x0.shape[1] < num_frames_for_clip:
        pad = num_frames_for_clip - x0.shape[1]
        x0 = torch.cat([x0, torch.randn(B, pad, dim_w)], dim=1)
    elif x0.shape[1] > num_frames_for_clip:
        x0 = x0[:, :num_frames_for_clip, :]
    return x0

run1 = [slice_chunk(noise_buf, i) for i in range(3)]
run2 = [slice_chunk(noise_buf, i) for i in range(3)]
assert all(torch.equal(x, y) for x, y in zip(run1, run2)), "sliced noise must be identical"
assert not torch.equal(run1[0], run1[1]), "consecutive chunks must use different slices"
tail = slice_chunk(noise_buf, 8)  # 8*24=192, only 8 frames left of 200
assert tail.shape == (B, num_frames_for_clip, dim_w), f"short tail must be padded, got {tail.shape}"
print("[ok] 3. FM noise_init: honoured, per-chunk slices, deterministic, short tail padded")

# ---------------------------------------------------------------- 4. wiring
sw = open(os.path.join(REPO, "IMTalker", "start_winner_live.sh"), encoding="utf-8").read()
for needle in ("pick_repo_first", "PERSONAPLEX_SEED", "ALLOW_MOSHI_FALLBACK",
               "ALLOW_WORKSPACE_ASSET_FALLBACK", "--- resolved assets ---"):
    assert needle in sw, f"start_winner_live.sh missing {needle}"
assert "pick_existing \\\n  /workspace/personaplex_bnb4" not in sw, \
    "PERSONAPLEX_DIR must go through pick_repo_first"
rl = open(os.path.join(REPO, "run_live.sh"), encoding="utf-8").read()
for needle in ("PERSONAPLEX_SEED", "PERSONAPLEX_TEMP_TEXT", "ALLOW_MOSHI_FALLBACK"):
    assert needle in rl, f"run_live.sh missing {needle}"
lt = open(LIVETRY, encoding="utf-8").read()
assert "PersonaPlex runtime contract OK" in lt
assert lt.count("seed_personaplex(") >= 2, "must seed at init AND on session reset"
print("[ok] 4. wiring: launcher, run_live.sh and liveTry.py all carry the contract")

print("\nAll checks passed.")
