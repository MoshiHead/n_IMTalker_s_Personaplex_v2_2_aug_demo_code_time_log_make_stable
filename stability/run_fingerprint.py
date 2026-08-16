#!/usr/bin/env python3
"""Fingerprint one PersonaPlex + IMTalker run so a working pod can be diffed
against a broken one.

Run it INSIDE the same venv the live server uses, BEFORE (or after) launching:

    source /workspace/preprocess_5090/bin/activate
    python stability/run_fingerprint.py --out /workspace/fp_$(date +%s).json

Then, with two files from two different pods:

    python stability/run_fingerprint.py --compare fp_good.json fp_bad.json

Everything captured here is something that can silently differ between two
"identical" pod runs and change what PersonaPlex actually is:

  * which `moshi` package got imported (the repo copy and the copy bundled in
    the bnb4 weights snapshot are DIFFERENT forks with different LMGen
    signatures -- liveTry.py silently falls back when the kwargs are rejected)
  * whether `--quantize_4bit` was honoured or silently dropped by the
    inspect.signature filter in liveTry.MoshiOnlyEngine.__init__
  * whether CFG condition tensors and the on_text_hook were wired up
  * the sampling defaults actually in force (temp / top_k)
  * unpinned dependency versions (bitsandbytes above all -- it is the 4-bit
    dequant kernel for the whole 7B model on sm_120)
  * the sha256 of every asset start_winner_live.sh will really resolve, using
    the same pick_existing precedence the launcher uses (which prefers stale
    /workspace/... copies over the fresh checkout)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# Assets big enough that hashing them fully costs real time. Head+tail+size is
# enough to tell two different checkpoints apart.
_PARTIAL_HASH_BYTES = 8 * 1024 * 1024


def _sha256(path: str, partial: bool = True) -> dict:
    p = Path(path)
    if not p.is_file():
        return {"exists": False}
    size = p.stat().st_size
    h = hashlib.sha256()
    with p.open("rb") as f:
        if partial and size > 2 * _PARTIAL_HASH_BYTES:
            h.update(f.read(_PARTIAL_HASH_BYTES))
            f.seek(-_PARTIAL_HASH_BYTES, os.SEEK_END)
            h.update(f.read(_PARTIAL_HASH_BYTES))
            kind = "head+tail"
        else:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
            kind = "full"
    return {"exists": True, "size": size, "sha256": h.hexdigest(), "hash_kind": kind}


def _pick_existing(*candidates: str) -> str:
    """Same precedence start_winner_live.sh's pick_existing() uses."""
    for c in candidates:
        for expanded in sorted(glob.glob(c)) or [c]:
            if os.path.exists(expanded):
                return expanded
    return ""


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in (
        "torch", "torchvision", "torchaudio", "numpy", "transformers", "peft",
        "bitsandbytes", "safetensors", "sentencepiece", "sphn", "einops",
        "huggingface_hub", "accelerate", "opencv-python",
    ):
        try:
            m = __import__(mod.replace("-", "_"))
            out[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001 - a missing module IS the datum
            out[mod] = f"<not importable: {type(e).__name__}>"
    return out


def _torch_cuda() -> dict:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        # The whole point of this report: without an explicit manual_seed the
        # default generator is seeded from OS entropy at process start, so this
        # value is DIFFERENT on every launch.
        "initial_seed_cpu": torch.initial_seed(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
        info["device_count"] = torch.cuda.device_count()
    return info


def _moshi_identity(personaplex_dir: str, root: str) -> dict:
    """Which moshi fork is importable, and what it actually supports.

    Mirrors liveTry._ensure_moshi_importable + the PYTHONPATH that
    start_winner_live.sh exports, so this resolves the same package the server
    will resolve. The repo's own personaplex/moshi is only probed as a fallback
    -- exactly the way notebook Step 11 falls back to it.
    """
    import inspect

    pkg = os.path.join(personaplex_dir, "moshi") if personaplex_dir else ""
    fallback = os.path.join(root, "personaplex", "moshi")
    out: dict = {
        "bnb4_moshi_path": pkg,
        "bnb4_moshi_exists": bool(pkg) and os.path.isdir(pkg),
        "repo_moshi_path": fallback,
        "repo_moshi_exists": os.path.isdir(fallback),
    }
    for candidate in (pkg, fallback):
        if candidate and os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
            break
    try:
        import moshi
        from moshi.models import LMGen, loaders
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
        return out

    out["module_file"] = getattr(moshi, "__file__", "?")
    out["version"] = getattr(moshi, "__version__", "?")

    lmgen_params = inspect.signature(LMGen.__init__).parameters
    out["lmgen_params"] = sorted(lmgen_params)
    # liveTry.py builds LMGen(lm, cfg_coef=..., condition_tensors=...,
    # on_text_hook=...) and falls back to LMGen(lm, device=...) on TypeError.
    # If these are False, that fallback fires: no CFG, no condition tensors,
    # and self.sampled_text is never populated.
    out["supports_cfg_coef"] = "cfg_coef" in lmgen_params
    out["supports_condition_tensors"] = "condition_tensors" in lmgen_params
    out["supports_on_text_hook"] = "on_text_hook" in lmgen_params
    out["lmgen_fallback_would_fire"] = not all(
        k in lmgen_params for k in ("cfg_coef", "condition_tensors", "on_text_hook")
    )
    out["sampling_defaults"] = {
        k: (lmgen_params[k].default if k in lmgen_params else "<absent>")
        for k in ("use_sampling", "temp", "temp_text", "top_k", "top_k_text",
                  "audio_silence_frame_cnt")
    }

    loader_params = inspect.signature(loaders.get_moshi_lm).parameters
    out["get_moshi_lm_params"] = sorted(loader_params)
    # liveTry filters optional kwargs through inspect.signature, so an
    # unsupported loader makes --quantize_4bit vanish with no error at all.
    out["quantize_4bit_honoured"] = "quantize_4bit" in loader_params
    out["num_codebooks_honoured"] = "num_codebooks" in loader_params
    out["context_honoured"] = "context" in loader_params

    # liveTry wraps this import in a bare `except Exception: cond_tensors = {}`.
    try:
        from moshi.run_inference import get_condition_tensors  # noqa: F401
        out["run_inference_importable"] = True
    except Exception as e:  # noqa: BLE001
        out["run_inference_importable"] = False
        out["run_inference_error"] = repr(e)

    try:
        out["lm_defaults"] = {
            k: loaders._lm_kwargs[k]
            for k in ("dim", "n_q", "dep_q", "num_layers", "context", "delays")
            if k in loaders._lm_kwargs
        }
    except Exception:  # noqa: BLE001
        pass
    return out


def _git(root: str) -> dict:
    def _run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", root, *args], capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "remote": _run("config", "--get", "remote.origin.url"),
        "dirty": bool(_run("status", "--porcelain")),
    }


def _pip_freeze() -> dict:
    try:
        txt = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=180
        ).stdout
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    lines = sorted(l.strip() for l in txt.splitlines() if l.strip())
    return {
        "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
        "count": len(lines),
        "packages": lines,
    }


def collect(root: str, full_hash: bool) -> dict:
    imtalker = os.path.join(root, "IMTalker")
    ckpt = os.path.join(root, "checkpoints")

    # Exactly the precedence start_winner_live.sh applies.
    personaplex_dir = _pick_existing(
        "/workspace/personaplex_bnb4",
        os.path.join(ckpt, "personaplex_bnb4"),
    )
    adapter = _pick_existing(
        "/workspace/hf_assets/personaplex_lookahead_rms_adapter/checkpoints/"
        "personaplex_lookahead096_future048_rms50_adapter.pt",
        os.path.join(ckpt, "personaplex_lookahead_rms_adapter/checkpoints/"
                           "personaplex_lookahead096_future048_rms50_adapter.pt"),
    )
    silence = _pick_existing(
        "/workspace/hf_assets/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt",
        "/workspace/personaplex_frontend_adapter_dataset/stats/silence_helium_mean.pt",
        os.path.join(ckpt, "personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt"),
    )
    lora = _pick_existing(
        os.path.join(ckpt, "live_winner/lora/"
                           "ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt"),
        os.path.join(imtalker, "checkpoints/ditto_blink_lora_withaudio_r64_1h_last.ckpt"),
        "/workspace/hf_assets/lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt",
        os.path.join(ckpt, "lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt"),
    )
    voice_name = os.environ.get("VOICE_PROMPT", "VARM3.pt")
    voice_dir = os.environ.get("VOICE_PROMPT_DIR", "") or _pick_existing(
        os.path.join(personaplex_dir, "voices"),
        "/workspace/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices",
        "/root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices",
        os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/"
                           "snapshots/*/voices"),
    )

    assets = {
        "personaplex_dir": personaplex_dir,
        "moshi_weight": os.path.join(personaplex_dir, "model_bnb_4bit.pt"),
        "mimi_weight": os.path.join(
            personaplex_dir, "tokenizer-e351c8d8-checkpoint125.safetensors"),
        "text_tokenizer": os.path.join(personaplex_dir, "tokenizer_spm_32k_3.model"),
        "voice_prompt": os.path.join(voice_dir, voice_name) if voice_dir else "",
        "adapter": adapter,
        "silence_helium": silence,
        "lora_generator": lora,
        "generator": os.path.join(imtalker, "checkpoints/generator.ckpt"),
        "renderer": os.path.join(imtalker, "checkpoints/renderer.ckpt"),
        "wav2vec": os.path.join(imtalker, "checkpoints/wav2vec2-base-960h/pytorch_model.bin"),
        "system_prompt": os.path.join(imtalker, "prompts/RB_Robert_System_Prompt_full.txt"),
        "ref_lora_weights": os.path.join(ckpt, "rag_lora/lora/adapter_model.safetensors"),
        "ref_lora_config": os.path.join(ckpt, "rag_lora/lora/adapter_config.json"),
    }
    asset_hashes = {
        name: ({"path": path, **_sha256(path, partial=not full_hash)} if path
               else {"path": "", "exists": False})
        for name, path in assets.items()
        if name != "personaplex_dir"
    }
    asset_hashes["personaplex_dir"] = {"path": personaplex_dir,
                                       "exists": bool(personaplex_dir)}

    env_keys = [
        "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "PYTHONPATH",
        "IMTALKER_PROMPT_STATE_CACHE", "TOKENIZERS_PARALLELISM", "HF_HOME",
        "HF_HUB_OFFLINE", "ENABLE_SEARCH", "WEB_SEARCH_ENABLED",
        "WEB_SEARCH_PROVIDER", "ROUTER_THRESHOLD", "ROUTER_RULES",
        "COMPRESSOR_MODEL", "PROMPT_SETTLE_SEC", "MAX_INPUT_BUFFER_SEC",
        "SUPPRESS_TEXT_DURING_SEARCH", "A_CFG_SCALE", "NFE", "VOICE_PROMPT",
        "DISABLE_LORA", "PERSONAPLEX_SEED",
    ]

    return {
        "schema": 1,
        "root": root,
        "git": _git(root),
        "versions": _versions(),
        "torch_cuda": _torch_cuda(),
        "moshi": _moshi_identity(personaplex_dir, root),
        "assets": asset_hashes,
        "env": {k: os.environ.get(k, "<unset>") for k in env_keys},
        "pip": _pip_freeze(),
    }


# Fields whose value differing between two runs is, on its own, enough to
# explain a change in answer quality. Reported first and loudly.
_CRITICAL = [
    ("moshi", "module_file"),
    ("moshi", "version"),
    ("moshi", "quantize_4bit_honoured"),
    ("moshi", "lmgen_fallback_would_fire"),
    ("moshi", "supports_cfg_coef"),
    ("moshi", "supports_condition_tensors"),
    ("moshi", "supports_on_text_hook"),
    ("moshi", "run_inference_importable"),
    ("moshi", "sampling_defaults"),
    ("versions", "bitsandbytes"),
    ("versions", "transformers"),
    ("versions", "peft"),
    ("versions", "torch"),
    ("versions", "numpy"),
    ("torch_cuda", "device_name"),
    ("torch_cuda", "capability"),
    ("git", "commit"),
]


def _flatten(d, prefix=""):
    flat = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key + "."))
        else:
            flat[key] = v
    return flat


def compare(path_a: str, path_b: str) -> int:
    a = json.load(open(path_a, encoding="utf-8"))
    b = json.load(open(path_b, encoding="utf-8"))

    print(f"A = {path_a}")
    print(f"B = {path_b}\n")

    print("== CRITICAL (any difference here explains a quality change) ==")
    critical_diffs = 0
    for section, key in _CRITICAL:
        va, vb = a.get(section, {}).get(key), b.get(section, {}).get(key)
        if va != vb:
            critical_diffs += 1
            print(f"  !! {section}.{key}\n       A: {va}\n       B: {vb}")
    if not critical_diffs:
        print("  (none -- the model, its fork, and its numerics are the same)")

    print("\n== ASSETS ==")
    asset_diffs = 0
    for name in sorted(set(a.get("assets", {})) | set(b.get("assets", {}))):
        va = a.get("assets", {}).get(name, {})
        vb = b.get("assets", {}).get(name, {})
        if va.get("sha256") != vb.get("sha256") or va.get("path") != vb.get("path"):
            asset_diffs += 1
            print(f"  !! {name}")
            print(f"       A: {va.get('path')} sha={va.get('sha256', '-')} "
                  f"exists={va.get('exists')}")
            print(f"       B: {vb.get('path')} sha={vb.get('sha256', '-')} "
                  f"exists={vb.get('exists')}")
    if not asset_diffs:
        print("  (identical)")

    print("\n== PACKAGES ==")
    pa = set(a.get("pip", {}).get("packages", []))
    pb = set(b.get("pip", {}).get("packages", []))
    only_a, only_b = sorted(pa - pb), sorted(pb - pa)
    if not only_a and not only_b:
        print("  (identical environments)")
    else:
        names_a = {p.split("==")[0].lower(): p for p in only_a}
        names_b = {p.split("==")[0].lower(): p for p in only_b}
        for name in sorted(set(names_a) | set(names_b)):
            print(f"  {name}: A={names_a.get(name, '<absent>')}  B={names_b.get(name, '<absent>')}")

    print("\n== ENV / OTHER ==")
    fa, fb = _flatten({k: v for k, v in a.items() if k in ("env", "torch_cuda")}), \
             _flatten({k: v for k, v in b.items() if k in ("env", "torch_cuda")})
    other = 0
    for key in sorted(set(fa) | set(fb)):
        # initial_seed_cpu differs on EVERY launch by design until the seed fix
        # lands; call that out rather than listing it as noise.
        if key.endswith("initial_seed_cpu"):
            if fa.get(key) != fb.get(key):
                print(f"  ~~ {key}: A={fa.get(key)} B={fb.get(key)}")
                print("       ^ unseeded RNG: PersonaPlex samples a different rollout each launch")
            continue
        if fa.get(key) != fb.get(key):
            other += 1
            print(f"  {key}: A={fa.get(key)}  B={fb.get(key)}")
    if not other:
        print("  (identical)")

    print(f"\nSummary: {critical_diffs} critical, {asset_diffs} asset, "
          f"{len(only_a) + len(only_b)} package differences.")
    return 1 if (critical_diffs or asset_diffs) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("SPEECH2AVATAR_ROOT",
                                                     "/workspace/speech2avatar"))
    ap.add_argument("--out", default="")
    ap.add_argument("--full-hash", action="store_true",
                    help="hash checkpoints in full instead of head+tail (slow)")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)

    fp = collect(args.root, args.full_hash)
    text = json.dumps(fp, indent=2, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[ok] wrote {args.out}")
    else:
        print(text)

    m = fp.get("moshi", {})
    print("\n--- headline ---", file=sys.stderr)
    if m.get("error"):
        print(f"moshi:            NOT IMPORTABLE -- {m['error']}", file=sys.stderr)
        print(f"                  bnb4 copy exists={m.get('bnb4_moshi_exists')} "
              f"repo copy exists={m.get('repo_moshi_exists')}", file=sys.stderr)
    else:
        print(f"moshi:            {m.get('module_file')}  (v{m.get('version')})", file=sys.stderr)
        print(f"quantize_4bit:    "
              f"{'HONOURED' if m.get('quantize_4bit_honoured') else 'SILENTLY DROPPED'}",
              file=sys.stderr)
        print(f"LMGen fallback:   "
              f"{'FIRED (no CFG, no condition tensors, no text hook)' if m.get('lmgen_fallback_would_fire') else 'not fired'}",
              file=sys.stderr)
    print(f"bitsandbytes:     {fp['versions'].get('bitsandbytes')}", file=sys.stderr)
    print(f"transformers:     {fp['versions'].get('transformers')}", file=sys.stderr)
    print(f"torch initial_seed: {fp['torch_cuda'].get('initial_seed_cpu')} "
          f"(random per launch until PersonaPlex is seeded)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
