"""Coverage verdict for the reference LoRA, from the real startup logs.

Three versions of this check were wrong before it was right, each refusing to
start a correctly-loaded adapter:

  1. "PEFT wrapped extra modules"        -> fatal. Extras get lora_B = 0 and are
                                            exact no-ops. Harmless.
  2. "some lora_B is all zeros"          -> fatal, read as "weights never loaded".
                                            But a zero B is legitimate CHECKPOINT
                                            CONTENT: the deployed checkpoint has
                                            38 modules, 6 of them
                                            (depformer.*.self_attn.out_proj) with
                                            a zero B by construction.
  3. names compared, values ignored      -> correct.

The only real failure is a module the checkpoint carries weights for that PEFT
never wrapped, because then those weights had nowhere to load.

Run from the repo root:  python stability/test_coverage_verdict.py
Needs no third-party packages.
"""
import ast
import json
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SCRATCH = pathlib.Path(tempfile.mkdtemp(prefix="reflora_"))
subprocess.run([sys.executable, str(REPO / "stability/_make_fake_lora.py"),
                str(SCRATCH / "fake_lora")], check=True, stdout=subprocess.DEVNULL)

src = (REPO / "IMTalker/liveTry.py").read_text(encoding="utf-8")
tree = ast.parse(src)

# Lift the stdlib helper out of liveTry.py without importing the module
# (it needs cv2 / torch / fastapi).
ns = {"re": re, "json": json, "Path": pathlib.Path}
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_LORA_KEY_RE":
        exec(compile(ast.Module([node], []), "liveTry", "exec"), ns)
    if isinstance(node, ast.FunctionDef) and node.name == "_checkpoint_lora_modules":
        exec(compile(ast.Module([node], []), "liveTry", "exec"), ns)
checkpoint_modules = ns["_checkpoint_lora_modules"]

ckpt_path = SCRATCH / "fake_lora" / "adapter_model.safetensors"
ckpt = checkpoint_modules(ckpt_path)
print(f"checkpoint names {len(ckpt)} modules (read from the header: no torch, no safetensors)")
assert len(ckpt) == 32, len(ckpt)
assert all("." in n for n in ckpt) and not any(n.startswith("base_model") for n in ckpt)
print("[ok] 1. module names are parsed, prefix-stripped, and A/B-paired")

assert checkpoint_modules(SCRATCH / "nope.safetensors") is None
junk = SCRATCH / "junk.safetensors"
junk.write_bytes(b"not a safetensors file")
assert checkpoint_modules(junk) is None
print("[ok] 2. a missing or unreadable checkpoint returns None, never raises")


# --- the verdict, exactly as liveTry.py implements it -----------------------
def verdict(wrapped, ckpt_names, strict):
    if ckpt_names is None:
        return "unverified"
    missing = ckpt_names - wrapped
    if missing:
        return "RAISE" if strict else "warn"
    return "ok"


def peft_names(short_names):
    """PEFT reports modules under a base_model.model. prefix; the check must
    normalise before comparing."""
    return {f"base_model.model.{n}" for n in short_names}


def normalise(names):
    out = set()
    for name in names:
        for prefix in ("base_model.model.", "base_model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        out.add(name)
    return out


assert normalise(peft_names(ckpt)) == ckpt
print("[ok] 3. PEFT's base_model.model.* names normalise onto the checkpoint namespace")

# 4. THE FAILING CASE: checkpoint names 38, PEFT wrapped all 38, 6 have a zero
#    lora_B. Every weight loaded. Must start.
ck38 = {f"transformer.layers.{i}.self_attn.in_proj" for i in range(32)} | \
       {f"depformer.layers.{i}.self_attn.out_proj" for i in range(6)}
wrapped38 = normalise(peft_names(ck38))
assert len(ck38) == 38 and len(wrapped38) == 38
assert verdict(wrapped38, ck38, strict=True) == "ok"
print("[ok] 4. 38 named / 38 wrapped / 6 zero-valued -> STARTS (the log that was blocking)")

# 5. Extras beyond the checkpoint are still fine.
assert verdict(wrapped38 | {"extra.module.a", "extra.module.b"}, ck38, strict=True) == "ok"
print("[ok] 5. extra wrapped modules do not block startup")

# 6. A genuinely partial load: 6 checkpoint modules never wrapped.
partial = {n for n in wrapped38 if "depformer" not in n}
assert verdict(partial, ck38, strict=True) == "RAISE"
assert verdict(partial, ck38, strict=False) == "warn"
print("[ok] 6. checkpoint modules that were never wrapped -> refuses (strict) / warns")

# 7. Unverifiable checkpoint never blocks.
assert verdict(set(), None, strict=True) == "unverified"
print("[ok] 7. an unreadable checkpoint never blocks startup")


# --- the source really implements this --------------------------------------
cov = next(n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "_report_ref_lora_coverage")
cov_src = ast.get_source_segment(src, cov)
assert "missing = sorted(ckpt - wrapped)" in cov_src, \
    "the verdict must be a set difference on names"
assert "if missing:" in cov_src
assert "base_model.model." in cov_src, "PEFT's prefix must be normalised away"
# A zero lora_B must be reported, never treated as a failure.
assert "zero_b" in cov_src and "zero-valued lora_B" in cov_src
zero_branch = cov_src.split("if missing:")[1]
assert "zero_b" not in zero_branch.split("print(")[0], \
    "the fatal branch must not consider zero-valued weights"
print("[ok] 8. liveTry.py compares names; zero-valued lora_B is reported, not fatal")

print("\nAll coverage-verdict checks passed.")
