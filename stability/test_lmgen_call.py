"""The LMGen call must be built FROM the signature, for every fork shape.

Reproduces the two real signatures seen so far, including the bnb4 one from the
live_server.log traceback:

    TypeError: LMGen.__init__() missing 1 required positional argument: 'device'

and asserts that an UNSET sampling env var passes nothing, so the build's own
default survives.
"""
import inspect
import os

FRAME_RATE_HZ = 12.5


# --- the two real signatures -------------------------------------------------

class BnB4LMGen:
    """The fork bundled in personaplex_bnb4: `device` REQUIRED, no cfg_coef /
    condition_tensors / on_text_hook."""
    def __init__(self, lm_model, device, use_sampling=True, temp=0.8,
                 temp_text=0.7, top_k=250, top_k_text=25, check=False,
                 audio_silence_frame_cnt=1, text_prompt_tokens=None,
                 sample_rate=32000, frame_rate=FRAME_RATE_HZ):
        self.seen = dict(device=device, temp=temp, temp_text=temp_text,
                         top_k=top_k, top_k_text=top_k_text)


class FullLMGen:
    """A fork that takes the conditioning API and no device."""
    def __init__(self, lm_model, cfg_coef=1.0, condition_tensors=None,
                 on_text_hook=None, temp=0.9, temp_text=0.6, top_k=100,
                 top_k_text=10):
        self.seen = dict(cfg_coef=cfg_coef, condition_tensors=condition_tensors,
                         on_text_hook=on_text_hook, temp=temp, temp_text=temp_text,
                         top_k=top_k, top_k_text=top_k_text)


# --- the logic now in liveTry.py --------------------------------------------

def build(LMGen, lm, device, cfg_coef, cond_tensors, on_text_hook, env):
    sig = inspect.signature(LMGen.__init__)
    params = set(sig.parameters)

    sampling_env = {
        "temp": ("PERSONAPLEX_TEMP", float),
        "temp_text": ("PERSONAPLEX_TEMP_TEXT", float),
        "top_k": ("PERSONAPLEX_TOP_K", int),
        "top_k_text": ("PERSONAPLEX_TOP_K_TEXT", int),
    }
    sampling_kwargs, effective = {}, {}
    for key, (env_name, cast) in sampling_env.items():
        param = sig.parameters.get(key)
        build_default = None if param is None else param.default
        raw = (env.get(env_name) or "").strip()
        if not raw:
            effective[key] = f"{build_default} (build default)"
            continue
        if param is None:
            effective[key] = "<not supported by this build>"
            continue
        sampling_kwargs[key] = cast(raw)
        effective[key] = f"{sampling_kwargs[key]} (override)"

    core = {"cfg_coef": cfg_coef, "condition_tensors": cond_tensors,
            "on_text_hook": on_text_hook}
    missing = sorted(k for k in core if k not in params)
    for k in missing:
        core.pop(k)

    kwargs = dict(core)
    kwargs.update(sampling_kwargs)
    if "device" in params:
        kwargs["device"] = device

    names = [n for n in sig.parameters if n != "self"]
    model_param = names[0] if names else None
    unmet = [
        n for n, p in sig.parameters.items()
        if n not in ("self", model_param)
        and p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        and n not in kwargs
    ]
    if unmet:
        raise RuntimeError(f"requires {unmet}")
    return LMGen(lm, **kwargs), effective, missing


LM, DEV = object(), "cuda:0"
HOOK = lambda t: None

# 1. The exact crash: bnb4 fork, nothing in the environment.
gen, eff, missing = build(BnB4LMGen, LM, DEV, 1.0, {}, HOOK, {})
assert gen.seen["device"] == DEV, "device must be supplied when the signature requires it"
assert missing == ["cfg_coef", "condition_tensors", "on_text_hook"]
print("[ok] 1. bnb4 fork constructs -- device supplied, three kwargs correctly dropped")

# 2. And its OWN defaults survive untouched when nothing is set.
assert gen.seen["temp"] == 0.8 and gen.seen["top_k_text"] == 25, gen.seen
assert all("build default" in v for v in eff.values()), eff
print("[ok] 2. unset env passes nothing -- the build's own sampling defaults survive")

# 3. A fork with DIFFERENT defaults must also keep them. This is the case that
#    a hardcoded '0.8' fallback would have silently changed.
gen2, eff2, missing2 = build(FullLMGen, LM, DEV, 1.13, {"description": 1}, HOOK, {})
assert gen2.seen["temp"] == 0.9 and gen2.seen["top_k"] == 100, gen2.seen
assert missing2 == []
assert gen2.seen["cfg_coef"] == 1.13 and gen2.seen["condition_tensors"] == {"description": 1}
print("[ok] 3. a fork with different defaults keeps them; its core kwargs are passed")

# 4. An explicit override reaches the model on both forks.
env = {"PERSONAPLEX_TEMP_TEXT": "0.4", "PERSONAPLEX_TOP_K_TEXT": "8"}
gen3, eff3, _ = build(BnB4LMGen, LM, DEV, 1.0, {}, HOOK, env)
assert gen3.seen["temp_text"] == 0.4 and gen3.seen["top_k_text"] == 8, gen3.seen
assert gen3.seen["temp"] == 0.8, "unset knobs must stay at the build default"
assert "override" in eff3["temp_text"] and "build default" in eff3["temp"]
print("[ok] 4. explicit overrides apply; unset knobs stay at the build default")

# 5. A signature with a required parameter we cannot supply is named, not guessed.
class WeirdLMGen:
    def __init__(self, lm_model, mystery_required, temp=0.8):
        ...

try:
    build(WeirdLMGen, LM, DEV, 1.0, {}, HOOK, {})
    raise AssertionError("must refuse a required parameter it cannot supply")
except RuntimeError as e:
    assert "mystery_required" in str(e), e
print("[ok] 5. an unknown required parameter is named in the error, not guessed at")

# 6. The real file must contain this logic, not the old hardcoded call.
src = open("IMTalker/liveTry.py", encoding="utf-8").read()
assert 'if "device" in _lmgen_params:' in src, "device must be supplied from the signature"
assert "self.lm_gen = LMGen(self.lm, **lmgen_kwargs)" in src
assert 'os.environ.get("PERSONAPLEX_TEMP", "0.8")' not in src, \
    "no hardcoded sampling fallback may remain"
assert "required_unmet" in src
print("[ok] 6. liveTry.py builds the call from the signature, with no hardcoded fallback")

print("\nAll LMGen construction checks passed.")
