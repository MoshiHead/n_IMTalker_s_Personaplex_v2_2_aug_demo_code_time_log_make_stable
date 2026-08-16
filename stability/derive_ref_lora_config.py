#!/usr/bin/env python3
"""Write an adapter_config.json that exactly matches the reference LoRA checkpoint.

Why this exists
---------------
`scripts/download_live_assets.sh` publishes only `adapter_model.safetensors` and
then HAND-WRITES a config with guessed `target_modules`:

    "target_modules": ["proj", "fc1", "out_proj", "fc2", "linear", "in_proj"]

PEFT treats those as SUFFIXES, so it wraps every module in the 7B whose name
ends with one of them -- far more modules than the checkpoint actually trained.
Each extra module is created with `init_lora_weights: true`, which means
lora_A random and **lora_B zero**, i.e. a delta of exactly zero. The startup log
measured the damage directly:

    reference LoRA coverage: 32/76 wrapped modules carry trained weights,
    44 are zero-initialised no-ops

and PEFT warned about missing keys such as
`base_model.model.depformer.layers.3.self_attn.in_proj.lora_A.default.weight`.

An adapter loaded against the wrong module set is not the adapter that was
trained. Since this one was trained specifically to teach the model to consume
injected reference blocks, a partial load is exactly why injected facts get
ignored.

This tool reads the checkpoint and writes the config that matches it: the exact
module paths that carry weights, and the rank read off the tensor shapes.

Usage
-----
    python stability/derive_ref_lora_config.py <lora_dir> [--alpha-ratio 2.0]
    python stability/derive_ref_lora_config.py <lora_dir> --check   # verify only

`<lora_dir>` is the directory holding `adapter_model.safetensors`
(normally `checkpoints/rag_lora/lora`).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# base_model.model.<path>.lora_A[.<adapter>].weight
_LORA_KEY_RE = re.compile(
    r"^(?:base_model\.model\.)?(?P<module>.+?)\.lora_(?P<ab>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$"
)


def parse_checkpoint(path: pathlib.Path) -> dict:
    try:
        from safetensors import safe_open
    except ImportError:
        sys.exit("safetensors is required: pip install safetensors")

    modules: dict[str, dict] = {}
    adapters: set[str] = set()
    skipped: list[str] = []
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            m = _LORA_KEY_RE.match(key)
            if not m:
                skipped.append(key)
                continue
            name = m.group("module")
            if m.group("adapter"):
                adapters.add(m.group("adapter"))
            shape = list(f.get_slice(key).get_shape())
            modules.setdefault(name, {})[m.group("ab")] = shape

    ranks: set[int] = set()
    complete, incomplete = [], []
    for name, parts in sorted(modules.items()):
        if "A" in parts and "B" in parts:
            complete.append(name)
            # lora_A is [r, in_features]; lora_B is [out_features, r]
            ranks.add(int(parts["A"][0]))
        else:
            incomplete.append(name)

    return {
        "path": str(path),
        "modules": complete,
        "incomplete_modules": incomplete,
        "adapters": sorted(adapters),
        "ranks": sorted(ranks),
        "unparsed_keys": skipped,
    }


def build_config(info: dict, alpha_ratio: float, alpha_override: float | None) -> dict:
    ranks = info["ranks"]
    if len(ranks) != 1:
        sys.exit(
            f"checkpoint mixes ranks {ranks}; a single adapter_config cannot describe it. "
            f"Use rank_pattern, or split the adapter."
        )
    r = ranks[0]
    alpha = float(alpha_override) if alpha_override is not None else float(r) * alpha_ratio
    return {
        "peft_type": "LORA",
        "task_type": "FEATURE_EXTRACTION",
        "base_model_name_or_path": None,
        "inference_mode": True,
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        # FULL module paths, not suffixes. PEFT matches a list entry with
        # `endswith`, so full paths wrap exactly the modules the checkpoint
        # trained -- no extra modules created as zero no-ops, no missing keys.
        "target_modules": info["modules"],
        "modules_to_save": None,
        "init_lora_weights": True,
        "use_rslora": False,
        "use_dora": False,
        "alpha_pattern": {},
        "rank_pattern": {},
        "layers_pattern": None,
        "layers_to_transform": None,
        "revision": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lora_dir", help="directory containing adapter_model.safetensors")
    ap.add_argument("--alpha-ratio", type=float, default=2.0,
                    help="lora_alpha = r * this (default 2.0, matching the previous "
                         "hand-written config's 256/128). Ignored if --alpha is given.")
    ap.add_argument("--alpha", type=float, default=None, help="set lora_alpha explicitly")
    ap.add_argument("--check", action="store_true",
                    help="report what the existing config would load, and exit non-zero "
                         "if it does not match the checkpoint")
    args = ap.parse_args()

    lora_dir = pathlib.Path(args.lora_dir)
    weights = lora_dir / "adapter_model.safetensors"
    if not weights.is_file():
        sys.exit(f"no adapter_model.safetensors in {lora_dir}")

    info = parse_checkpoint(weights)
    print(f"checkpoint: {weights}")
    print(f"  trained modules : {len(info['modules'])}")
    print(f"  rank            : {info['ranks']}")
    if info["adapters"]:
        print(f"  adapter names   : {info['adapters']}")
    if info["incomplete_modules"]:
        print(f"  [warn] {len(info['incomplete_modules'])} module(s) have only one of "
              f"lora_A/lora_B: {info['incomplete_modules'][:5]}")
    if info["unparsed_keys"]:
        print(f"  [warn] {len(info['unparsed_keys'])} key(s) not recognised as LoRA: "
              f"{info['unparsed_keys'][:5]}")
    if not info["modules"]:
        sys.exit("no complete lora_A/lora_B pairs found -- is this a LoRA checkpoint?")

    # Group the module paths by their last component, purely for a readable summary.
    by_suffix: dict[str, int] = {}
    for name in info["modules"]:
        by_suffix[name.rsplit(".", 1)[-1]] = by_suffix.get(name.rsplit(".", 1)[-1], 0) + 1
    print(f"  by module type  : {by_suffix}")

    config_path = lora_dir / "adapter_config.json"
    new_config = build_config(info, args.alpha_ratio, args.alpha)

    if args.check:
        if not config_path.is_file():
            print(f"\n[FAIL] no adapter_config.json at {config_path}")
            return 1
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        problems = []
        targets = existing.get("target_modules")
        if isinstance(targets, list) and set(targets) != set(info["modules"]):
            suffix_style = all("." not in t for t in targets)
            problems.append(
                f"target_modules {'is a SUFFIX list' if suffix_style else 'does not match'} "
                f"({len(targets)} entries) -- PEFT will wrap modules the checkpoint never "
                f"trained, and each becomes a zero no-op"
            )
        if int(existing.get("r", -1)) != info["ranks"][0]:
            problems.append(f"r={existing.get('r')} but the checkpoint has r={info['ranks'][0]}")
        if problems:
            print("\n[FAIL] the existing config does not match the checkpoint:")
            for p in problems:
                print(f"  - {p}")
            print(f"\nFix it:  python {sys.argv[0]} {lora_dir}")
            return 1
        print("\n[ok] the existing adapter_config.json matches the checkpoint")
        return 0

    if config_path.is_file():
        backup = lora_dir / "adapter_config.json.bak"
        backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\n  previous config backed up to {backup}")
    config_path.write_text(json.dumps(new_config, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {config_path}")
    print(f"     r={new_config['r']} lora_alpha={new_config['lora_alpha']} "
          f"(scaling {new_config['lora_alpha'] / new_config['r']:.2f}) "
          f"target_modules={len(new_config['target_modules'])} exact paths")
    print("\nNext startup should log 'reference LoRA coverage: N/N' with zero no-ops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
