"""Runtime-contract tests for the PersonaPlex startup path.

Two things are checked, both derived from a real startup failure:

  A. condition tensors -- the PersonaPlex fork bundled in the bnb4 weights
     snapshot ships no `moshi.run_inference`, and that fork is the CORRECT one.
     A missing import there must not be treated as a fault; only a model that
     DECLARES conditioners nobody can build for it is.

  B. the strict gates -- a kwarg an LMGen or loader build does not accept is
     only a fault when its value would have changed something. Failing on mere
     absence is just a new way to break a healthy deployment.

Run:  python stability/test_runtime_contract.py      (from the repo root)
Needs only torch.
"""

print("-- part A: condition tensors --")

import ast
import contextlib
import sys
import traceback
import types

import torch

LIVETRY = "IMTalker/liveTry.py"

# Lift the two methods off MoshiOnlyEngine without importing the module (which
# needs cv2/torchaudio/fastapi/moshi).
tree = ast.parse(open(LIVETRY, encoding="utf-8").read())
cls = next(n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == "MoshiOnlyEngine")
wanted = {"_model_conditioners", "_resolve_condition_tensors"}
methods = {}
ns = {"contextlib": contextlib, "traceback": traceback, "torch": torch,
      "__import__": __import__, "print": lambda *a, **k: None}
for node in cls.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        exec(compile(ast.Module([node], []), LIVETRY, "exec"), ns)
        methods[node.name] = ns[node.name]
assert set(methods) == wanted, f"missing {wanted - set(methods)}"


class Engine:
    _model_conditioners = methods["_model_conditioners"]
    _resolve_condition_tensors = methods["_resolve_condition_tensors"]

    def __init__(self, lm, strict=True):
        self.lm = lm
        self.strict_runtime = strict
        self._runtime_contract = {"moshi_file": "/fake/personaplex_bnb4/moshi/moshi/__init__.py"}
        self.conv_logger = types.SimpleNamespace(error=lambda *a, **k: None)


def fresh_moshi(with_run_inference, conditioners_on_model):
    """Install a fake `moshi` package matching a given fork's shape."""
    for name in [m for m in sys.modules if m == "moshi" or m.startswith("moshi.")]:
        del sys.modules[name]
    pkg = types.ModuleType("moshi")
    pkg.__path__ = []
    sys.modules["moshi"] = pkg
    for sub in ("models", "models.loaders", "models.lm", "conditioners", "utils"):
        mod = types.ModuleType(f"moshi.{sub}")
        sys.modules[f"moshi.{sub}"] = mod
        setattr(pkg, sub.split(".")[0], sys.modules["moshi.models"] if "." in sub else mod)
    if with_run_inference:
        ri = types.ModuleType("moshi.run_inference")
        ri.get_condition_tensors = lambda mt, lm, batch_size=1, cfg_coef=1.0: {
            "description": torch.zeros(batch_size, 4)
        }
        sys.modules["moshi.run_inference"] = ri
        pkg.run_inference = ri

    lm = types.SimpleNamespace()
    if conditioners_on_model:
        lm.condition_provider = types.SimpleNamespace(
            conditioners={"description": object(), "speaker": object()}
        )
    return lm


# 1. The bnb4 fork from the traceback: no run_inference, no conditioners on the
#    model. This MUST be healthy -- it is what crashed the server.
lm = fresh_moshi(with_run_inference=False, conditioners_on_model=False)
tensors, status = Engine(lm, strict=True)._resolve_condition_tensors("personaplex", 1.0, True)
assert (tensors, status) == ({}, "none-declared"), (tensors, status)
print("[ok] 1. bnb4 fork (no run_inference, no conditioners) -> none-declared, no crash")

# 2. A fork that ships the helper: tensors get built.
lm = fresh_moshi(with_run_inference=True, conditioners_on_model=True)
tensors, status = Engine(lm, strict=True)._resolve_condition_tensors("personaplex", 1.13, True)
assert status == "built" and len(tensors) == 1, (tensors, status)
print("[ok] 2. fork with moshi.run_inference -> built")

# 3. Genuine fault: the model HAS conditioners but no helper can build them.
#    Strict must refuse; the escape hatch must let it through.
lm = fresh_moshi(with_run_inference=False, conditioners_on_model=True)
try:
    Engine(lm, strict=True)._resolve_condition_tensors("personaplex", 1.0, True)
    raise AssertionError("strict mode must refuse conditioners it cannot build")
except RuntimeError as e:
    assert "declares conditioners" in str(e)
tensors, status = Engine(lm, strict=False)._resolve_condition_tensors("personaplex", 1.0, True)
assert (tensors, status) == ({}, "missing-helper"), (tensors, status)
print("[ok] 3. conditioners present but unbuildable -> refused when strict, allowed when not")

# 4. An LMGen that takes no condition_tensors at all: not applicable, not a fault.
lm = fresh_moshi(with_run_inference=False, conditioners_on_model=True)
tensors, status = Engine(lm, strict=True)._resolve_condition_tensors("personaplex", 1.0, False)
assert (tensors, status) == ({}, "unsupported-api"), (tensors, status)
print("[ok] 4. LMGen without a condition_tensors argument -> unsupported-api, no crash")

# 5. contract_ok must treat none-declared / unsupported-api as healthy and only
#    missing-helper as a failure.
src = open(LIVETRY, encoding="utf-8").read()
assert 'self._runtime_contract.get("condition_source") != "missing-helper"' in src
print("[ok] 5. contract_ok fails only on missing-helper")

print("\nAll condition-tensor checks passed.")

def loader_gate(optional_kwargs, supported):
    dropped = sorted(k for k in optional_kwargs if k not in supported)
    fatal = [
        k for k in dropped
        if optional_kwargs[k] not in (None, False)
        and not (k == "num_codebooks" and optional_kwargs[k] == 8)
    ]
    return dropped, fatal


def lmgen_gate(cfg_coef, cond_tensors, lmgen_params):
    core = {"cfg_coef": cfg_coef, "condition_tensors": cond_tensors, "on_text_hook": object()}
    missing = sorted(k for k in core if k not in lmgen_params)
    fatal, benign = [], []
    for k in missing:
        if k == "cfg_coef" and abs(float(cfg_coef) - 1.0) > 1e-6:
            fatal.append(k)
        elif k == "condition_tensors" and cond_tensors:
            fatal.append(k)
        else:
            benign.append(k)
    return missing, fatal, benign


FULL_LOADER = {"filename", "device", "dtype", "quantize_4bit", "num_codebooks", "context"}
OLD_LOADER = {"filename", "copy_missing_weights", "device", "dtype", "delays", "cpu_offload"}
FULL_LMGEN = {"self", "lm_model", "cfg_coef", "condition_tensors", "on_text_hook", "temp"}
NOHOOK_LMGEN = {"self", "lm_model", "device", "temp", "top_k"}

# 1. The live pod: quantize_4bit honoured (the traceback got past this gate).
dropped, fatal = loader_gate(
    {"quantize_4bit": True, "num_codebooks": 8, "context": None}, FULL_LOADER)
assert (dropped, fatal) == ([], []), (dropped, fatal)
print("[ok] 1. loader gate silent on the bnb4 fork")

# 2. The repo fork: quantize_4bit=True silently discarded -> must be fatal.
dropped, fatal = loader_gate(
    {"quantize_4bit": True, "num_codebooks": 8, "context": None}, OLD_LOADER)
assert fatal == ["quantize_4bit"], fatal
assert dropped == ["context", "num_codebooks", "quantize_4bit"], dropped
print("[ok] 2. loader gate fatal only on the dropped quantize_4bit, not on context=None")

# 3. quantize_4bit not requested: dropping it costs nothing.
dropped, fatal = loader_gate(
    {"quantize_4bit": False, "num_codebooks": 8, "context": None}, OLD_LOADER)
assert fatal == [], fatal
print("[ok] 3. loader gate silent when the dropped values are all defaults")

# 4. The case that crashed the server: no condition tensors, cfg_coef 1.0,
#    LMGen without the hook. Nothing is lost -> must NOT be fatal.
missing, fatal, benign = lmgen_gate(1.0, {}, NOHOOK_LMGEN)
assert fatal == [], fatal
assert set(benign) == {"cfg_coef", "condition_tensors", "on_text_hook"}, benign
print("[ok] 4. LMGen gate quiet at cfg_coef=1.0 with no condition tensors")

# 5. A CFG scale WAS requested and would be silently ignored -> fatal.
missing, fatal, benign = lmgen_gate(1.13, {}, NOHOOK_LMGEN)
assert fatal == ["cfg_coef"], fatal
print("[ok] 5. LMGen gate fatal when a requested cfg_coef would be ignored")

# 6. Tensors were built but cannot be handed over -> fatal.
missing, fatal, benign = lmgen_gate(1.0, {"description": 1}, NOHOOK_LMGEN)
assert fatal == ["condition_tensors"], fatal
print("[ok] 6. LMGen gate fatal when built condition tensors cannot be passed")

# 7. The full fork: nothing missing at all.
missing, fatal, benign = lmgen_gate(1.13, {"description": 1}, FULL_LMGEN)
assert (missing, fatal, benign) == ([], [], []), (missing, fatal, benign)
print("[ok] 7. LMGen gate silent on the full PersonaPlex fork")

# 8. The launcher's real default must land in the safe case: start_winner_live.sh
#    never passes --moshi_cfg_coef, and the parser default is 1.0.
src = open("IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py",
           encoding="utf-8").read()
assert 'parser.add_argument("--moshi_cfg_coef", type=float, default=1.0)' in src
launcher = open("IMTalker/start_winner_live.sh", encoding="utf-8").read()
assert "--moshi_cfg_coef" not in launcher
print("[ok] 8. launcher leaves moshi_cfg_coef at 1.0, so case 4 is the real deployment")

print("\nAll gate checks passed.")
