"""Regression tests for the reference-LoRA load path (conversation_logs_5).

Two things went wrong, and they compound:

  1. The adapter was disabled by default. It is the component that teaches the
     model to ACT on an injected reference block, so without it a retrieved
     fact is just text the model said to itself.

  2. Even when enabled it was loaded against a HAND-WRITTEN config with guessed
     suffix-style target_modules, so PEFT wrapped 76 modules for a checkpoint
     that trained 32. The other 44 were created with lora_B = 0 -- exact
     no-ops -- while the trained modules went unmatched. An adapter applied to
     the wrong module set is not the adapter that was trained.

conversation_logs_5 also proved the pod was running older code than the
checkout (`ref_lora_loaded=False` alongside injected `<ref>` tags, which the
current revision cannot produce), so a revision marker is asserted here too.

Run from the repo root:  python stability/test_ref_lora.py
Needs no third-party packages.
"""
import ast
import json
import pathlib
import re
import sys

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
LIVETRY = REPO / "IMTalker/liveTry.py"
SERVER = REPO / "IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
LAUNCHER = REPO / "IMTalker/start_winner_live.sh"
TOOL = REPO / "stability/derive_ref_lora_config.py"

livetry_src = LIVETRY.read_text(encoding="utf-8")
server_src = SERVER.read_text(encoding="utf-8")
launcher = LAUNCHER.read_text(encoding="utf-8")


# ================================================ part A: enabled and strict
print("-- part A: the adapter is on, and must load completely --")

assert 'REF_LORA_ENABLED="${REF_LORA_ENABLED:-1}"' in launcher, \
    "the adapter must default ON -- it is what makes injection work"
assert 'REF_LORA_STRICT="${REF_LORA_STRICT:-1}"' in launcher
assert "--ref_lora_strict" in launcher and "--ref_lora_strict" in server_src
print("[ok] 1. REF_LORA_ENABLED defaults to 1, REF_LORA_STRICT to 1")

# A partial load must raise, and must NOT be swallowed by the load try/except.
tree = ast.parse(livetry_src)
load_fn = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_load_ref_lora")
load_src = ast.get_source_segment(livetry_src, load_fn)
try_nodes = [n for n in ast.walk(load_fn) if isinstance(n, ast.Try)]
guarded = set()
for t in try_nodes:
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            guarded.add(n.func.attr)
assert "_report_ref_lora_coverage" not in guarded, (
    "the coverage check must sit OUTSIDE the try/except, or a strict failure is "
    "caught and downgraded to 'continuing without it'"
)
assert "_report_ref_lora_coverage" in load_src
print("[ok] 2. the coverage verdict is outside the try, so strict mode can stop startup")

cov_fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_report_ref_lora_coverage")
cov_src = ast.get_source_segment(livetry_src, cov_fn)
assert "ref_lora_strict" in cov_src and "raise RuntimeError" in cov_src
assert "derive_ref_lora_config.py" in cov_src, "the error must name the fix"
print("[ok] 3. a partial load raises, and the message points at the repair tool")


# ============================== part B: requested != loaded, and tags follow
print("\n-- part B: the status must reflect reality --")

assert "self.ref_lora_active = False" in livetry_src
assert "ref_lora_loaded=bool(self.ref_lora_active)" in livetry_src, \
    "component_status must report the ACTUAL load, not merely that a dir was passed"
assert "ref_lora_requested=bool(self.ref_lora_dir)" in livetry_src
print("[ok] 4. component_status separates 'requested' from 'actually loaded'")

# The tag decision must follow the real state. logs_5 showed
# ref_lora_loaded=False together with injected <ref> tags.
assert "if self.ref_lora_active else ref_content" in server_src
assert 'if self.ref_lora_active else "Please wait a minute."' in server_src
assert "if self.ref_lora_dir else ref_content" not in server_src
print("[ok] 5. <ref>/<lookup> tags are used only when the adapter really loaded")


# ================================================= part C: the repair tool
print("\n-- part C: the config generator --")

assert TOOL.is_file(), "stability/derive_ref_lora_config.py must exist"
tool_src = TOOL.read_text(encoding="utf-8")
ast.parse(tool_src)

ns = {"re": re}
m = re.search(r"_LORA_KEY_RE = re\.compile\(\s*\n\s*(r\".*?\")\s*\n\)", tool_src, re.S)
assert m, "the key parser must be present"
exec("_LORA_KEY_RE = re.compile(" + m.group(1) + ")", ns)
R = ns["_LORA_KEY_RE"]

# Real key shapes, including the one PEFT named in its missing-keys warning.
for key, module, ab in [
    ("base_model.model.depformer.layers.3.self_attn.in_proj.lora_A.default.weight",
     "depformer.layers.3.self_attn.in_proj", "A"),
    ("base_model.model.transformer.layers.0.self_attn.out_proj.lora_B.default.weight",
     "transformer.layers.0.self_attn.out_proj", "B"),
    ("transformer.layers.7.gating.linear_in.lora_A.weight",
     "transformer.layers.7.gating.linear_in", "A"),
]:
    g = R.match(key)
    assert g and g.group("module") == module and g.group("ab") == ab, key
assert R.match("base_model.model.transformer.layers.0.self_attn.in_proj.weight") is None
print("[ok] 6. the key parser handles adapter-named and plain LoRA keys, ignores others")

# The generated config must use FULL paths, not suffixes -- that is the fix.
build = next(n for n in ast.walk(ast.parse(tool_src))
             if isinstance(n, ast.FunctionDef) and n.name == "build_config")
build_src = ast.get_source_segment(tool_src, build)
assert '"target_modules": info["modules"]' in build_src, \
    "target_modules must be the exact module paths from the checkpoint"
print("[ok] 7. it emits exact module paths, so PEFT wraps only trained modules")

assert "--check" in tool_src and "adapter_config.json.bak" in tool_src
print("[ok] 8. it can verify without writing, and backs up the previous config")

# Standard library only. The launcher runs this BEFORE activating the venv, so
# any third-party import makes the pre-flight check fail and, under
# REF_LORA_STRICT, stops the server from starting at all -- which is exactly
# what "safetensors is required: pip install safetensors" did.
imports = set()
for n in ast.walk(ast.parse(tool_src)):
    if isinstance(n, ast.Import):
        imports |= {a.name.split(".")[0] for a in n.names}
    elif isinstance(n, ast.ImportFrom) and n.module:
        imports.add(n.module.split(".")[0])
non_stdlib = sorted(m for m in imports if m not in sys.stdlib_module_names)
assert not non_stdlib, (
    f"the tool must import only the standard library, found {non_stdlib}; it runs "
    f"before the venv is active"
)
assert "read_safetensors_header" in tool_src, \
    "the safetensors header must be parsed directly, not via the package"
print(f"[ok] 9. stdlib-only ({', '.join(sorted(imports))}) -- runs before the venv exists")

# The launcher must run the check with an interpreter it can actually reach.
assert "derive_ref_lora_config.py" in launcher and "--check" in launcher
assert 'REF_LORA_PY="$VENV_DIR/bin/python"' in launcher, (
    "the launcher must resolve the venv interpreter explicitly -- this block runs "
    "before `source .../activate`, so a bare `python` is the system one"
)
assert '"$REF_LORA_PY" "$PROJECT_ROOT/stability/derive_ref_lora_config.py"' in launcher
print("[ok] 10. the launcher verifies the config on every start, using the venv python")


# ============================================== part D: the revision marker
print("\n-- part D: know which code is running --")

m = re.search(r'PIPELINE_REVISION\s*=\s*"([^"]+)"', livetry_src)
assert m, "liveTry.py must declare PIPELINE_REVISION"
rev = m.group(1)
assert "PIPELINE REVISION" in livetry_src, "it must be printed at startup"

nb = json.loads((REPO / "RunPod_RTX5090_PersonaPlex_IMTalker_Live_fixed.ipynb")
                .read_text(encoding="utf-8"))
nb_src = "\n".join("".join(c["source"]) for c in nb["cells"])
m2 = re.search(r'EXPECTED_PIPELINE_REVISION\s*=\s*"([^"]+)"', nb_src)
assert m2, "the notebook must pin an expected revision"
assert m2.group(1) == rev, (
    f"notebook expects {m2.group(1)!r} but liveTry.py declares {rev!r} -- "
    f"bump both together"
)
assert 'f"PIPELINE REVISION {EXPECTED_PIPELINE_REVISION}"' in nb_src, \
    "the health check must assert the revision"
print(f"[ok] 10. revision {rev} is printed, pinned, and health-checked")

assert "Step 11d" in nb_src and "derive_ref_lora_config" in nb_src
assert 'REF_LORA_ENABLED = "1"' in nb_src and 'REF_LORA_STRICT = "1"' in nb_src
print("[ok] 11. the notebook enables the adapter and regenerates its config at Step 11d")

print("\nAll reference-LoRA checks passed.")
