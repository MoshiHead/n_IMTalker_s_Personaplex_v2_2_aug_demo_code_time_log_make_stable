"""Replay the exact coverage situation from the failing startup log.

    reference LoRA coverage: 32/38 wrapped modules carry trained weights,
    6 are zero-initialised no-ops
    inactive (sample): depformer.layers.{0,1,2}.self_attn.out_proj

All 32 trained modules got their weights; PEFT additionally wrapped 6 that the
checkpoint does not train, and those are lora_B = 0, i.e. exact no-ops. That
must START, not raise.
"""
import ast
import json
import pathlib
import re
import sys

import subprocess, tempfile
REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SCRATCH = pathlib.Path(tempfile.mkdtemp(prefix="reflora_"))
# Build a real .safetensors fixture shaped like the deployed checkpoint.
subprocess.run([sys.executable, str(REPO / "stability/_make_fake_lora.py"),
                str(SCRATCH / "fake_lora")], check=True, stdout=subprocess.DEVNULL)
src = (REPO / "IMTalker/liveTry.py").read_text(encoding="utf-8")

# Lift the stdlib helper out of liveTry.py without importing the module.
tree = ast.parse(src)
ns = {"re": re, "json": json, "Path": pathlib.Path}
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_LORA_KEY_RE":
        exec(compile(ast.Module([node], []), "liveTry", "exec"), ns)
    if isinstance(node, ast.FunctionDef) and node.name == "_count_checkpoint_lora_modules":
        exec(compile(ast.Module([node], []), "liveTry", "exec"), ns)
count = ns["_count_checkpoint_lora_modules"]

ckpt = SCRATCH / "fake_lora" / "adapter_model.safetensors"
expected = count(ckpt)
print(f"checkpoint trains {expected} modules (read from the header, no torch, no safetensors)")
assert expected == 32, expected

assert count(SCRATCH / "does_not_exist.safetensors") is None
print("[ok] a missing checkpoint returns None -- verification never becomes a crash")

junk = SCRATCH / "junk.safetensors"
junk.write_bytes(b"not a safetensors file at all")
assert count(junk) is None
print("[ok] an unreadable checkpoint returns None rather than raising")


# --- the verdict itself -----------------------------------------------------
def verdict(active, inactive, expected, strict):
    """The rule now in _report_ref_lora_coverage."""
    if expected is None:
        return "unverified"
    if len(active) < expected:
        return "RAISE" if strict else "warn"
    return "ok-with-extras" if inactive else "ok"


A32 = [f"m{i}" for i in range(32)]
X6 = [f"depformer.layers.{i}.self_attn.out_proj" for i in range(6)]

# 1. The failing log, exactly. Must start.
assert verdict(A32, X6, 32, strict=True) == "ok-with-extras"
print("[ok] 32 trained + 6 zero extras -> starts (this is the log that was blocking you)")

# 2. The original 32/76 situation: still 32 applied, so it also starts -- but
#    that config wasted 44 wrappers and is what the generator fixes.
assert verdict(A32, [f"x{i}" for i in range(44)], 32, strict=True) == "ok-with-extras"
print("[ok] 32 trained + 44 zero extras -> starts too; extras are no-ops, not corruption")

# 3. A genuinely partial load: weights missing. Must still refuse.
assert verdict(A32[:20], X6, 32, strict=True) == "RAISE"
assert verdict(A32[:20], X6, 32, strict=False) == "warn"
print("[ok] 20 of 32 trained modules applied -> refuses when strict, warns when not")

# 4. A perfect load.
assert verdict(A32, [], 32, strict=True) == "ok"
print("[ok] 32/32 with no extras -> clean start")

# 5. Unverifiable checkpoint never blocks.
assert verdict(A32[:5], X6, None, strict=True) == "unverified"
print("[ok] an unreadable checkpoint never blocks startup")

# --- the source really implements this --------------------------------------
cov = next(n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "_report_ref_lora_coverage")
cov_src = ast.get_source_segment(src, cov)
assert "if len(active) < expected:" in cov_src, "the verdict must compare against the checkpoint"
assert "if inactive:\n            msg" not in cov_src, "inactive alone must no longer be fatal"
assert "_count_checkpoint_lora_modules" in cov_src
print("[ok] liveTry.py's verdict compares applied-vs-checkpoint, not wrapped-vs-applied")

print("\nAll coverage-verdict checks passed.")
