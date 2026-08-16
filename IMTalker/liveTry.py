"""liveTry.py - v3 one-websocket server, Step 3: Moshi audio/text only.

What this version does:
    browser mic PCM -> /ws/conversation -> original Moshi/Mimi
    Moshi reply audio/text -> JSON chunk_audio + static JPEG chunk_frame

What this version deliberately does NOT do yet:
    no VAD
    no Helium extraction
    no FM
    no IMTalker renderer
    no WebRTC / TURN / H264 / Opus

The goal is to prove the teammate-style HTML protocol works cleanly with our
original Moshi backend before adding Helium/IMTalker.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import sys
import tarfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


TARGET_SR = 24000
FRAME_SIZE = 1920  # 80 ms at 24 kHz, Moshi/Mimi step size

# Bump on every behavioural change to this file or the AH server. Printed at
# startup and recorded in the conversation log, because a pod running older
# code than you think is otherwise indistinguishable from a broken fix -- and
# was: conversation_logs_5 showed `ref_lora_loaded=False` alongside injected
# `<ref>` tags, a combination this revision cannot produce.
PIPELINE_REVISION = "2026-08-16.r12-bounded-forced-runs"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# LMGen samples EVERY text and audio token from the global torch RNG:
# moshi/utils/sampling.py calls `.exponential_(1, generator=None)`, which reads
# the default CUDA generator. Unseeded, PyTorch initialises that generator from
# OS entropy at process start, so each launch produced a different rollout of
# the same model -- and because the KV cache is never reset per turn (see
# reset_session), one unlucky rollout right after the system prompt coloured the
# whole session. That is what made "same pod, same question" land anywhere
# between a correct answer and unrelated speech.
#
# The `--seed` flag that start_winner_live.sh passes never reached this code:
# it is read only by generator/FM.py, and only when --fix_noise_seed is also
# set. So this is the first place PersonaPlex generation is seeded at all.

def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_personaplex_seed() -> int | None:
    """Seed for PersonaPlex generation. Empty PERSONAPLEX_SEED means 'stay
    random', which is the pre-fix behaviour and must remain reachable."""
    raw = os.environ.get("PERSONAPLEX_SEED", "42").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print(
            f"[liveTry] PERSONAPLEX_SEED={raw!r} is not an integer -- falling back to 42",
            flush=True,
        )
        return 42


def seed_personaplex(seed: int | None, where: str) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[liveTry] PersonaPlex sampling seeded: {seed} ({where})", flush=True)


_LORA_KEY_RE = re.compile(
    r"^(?:base_model\.model\.)?(?P<module>.+?)\.lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight$"
)


def _checkpoint_lora_modules(safetensors_path: Path) -> set[str] | None:
    """The module NAMES the LoRA checkpoint carries weights for.

    Names, not values, are what a load has to be judged on. Whether a module's
    lora_B happens to be all zeros is a property of how the adapter was TRAINED
    -- a zero B means that module contributes nothing, which is a legitimate
    thing for a checkpoint to contain. It says nothing about whether the load
    succeeded, and treating it as evidence of a failed load refuses to start a
    perfectly good adapter.

    What a load must guarantee is that every module the checkpoint names got
    wrapped, so its weights had somewhere to go.

    Reads the .safetensors header directly -- 8-byte little-endian length then
    JSON -- so it needs nothing beyond the standard library and never loads the
    weights. Returns None if the file cannot be read, so a verification failure
    never becomes a startup failure.
    """
    try:
        with safetensors_path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            n = int.from_bytes(raw, "little")
            if not 0 < n < (256 << 20):
                return None
            header = json.loads(f.read(n).decode("utf-8"))
    except Exception:
        return None

    header.pop("__metadata__", None)
    parts: dict[str, set[str]] = {}
    for key in header:
        m = _LORA_KEY_RE.match(key)
        if m:
            parts.setdefault(m.group("module"), set()).add(m.group("ab"))
    return {name for name, sides in parts.items() if {"A", "B"} <= sides}


def _ensure_moshi_importable(moshi_root: str | Path) -> None:
    root = Path(moshi_root)
    pkg = root / "moshi"
    if pkg.exists() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def _clean_text_piece(piece: str) -> str:
    return piece.replace("▁", " ")


def _make_placeholder_jpeg(path: str | Path | None) -> str:
    img = None
    if path:
        p = Path(path)
        if p.is_file():
            img = cv2.imread(str(p))
    if img is None:
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "Moshi",
            (150, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (235, 235, 235),
            3,
            cv2.LINE_AA,
        )
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise RuntimeError("failed to encode placeholder JPEG")
    return base64.b64encode(enc.tobytes()).decode("ascii")


class MoshiOnlyEngine:
    # RMS below which a 20ms window of microphone audio counts as a gap between
    # words rather than speech. Trimming those costs latency and nothing else.
    # Well under the 0.02 the server's own [MIC] logging calls "VOICE".
    _INPUT_SILENCE_RMS = 0.005
    # Multiple of max_input_buffer_sec at which unbounded delay becomes the
    # worse failure and speech may be cut after all.
    _INPUT_HARD_CEILING_FACTOR = 2.5

    def __init__(
        self,
        *,
        moshi_root: str,
        mimi_hf_repo: str,
        device: str,
        cfg_coef: float,
        placeholder_jpeg_b64: str,
        moshi_weight: str = "",
        mimi_weight: str = "",
        tokenizer: str = "",
        quantize_4bit: bool = False,
        num_codebooks: int = 8,
        context: int | None = None,
        voice_prompt: str = "",
        voice_prompt_dir: str = "",
        text_prompt: str = "",
        # --- STT + query routing + web search (all optional, off by default) ---
        ref_lora_dir: str = "",
        merge_ref_lora: bool = False,
        ref_lora_scale: float = 1.0,
        ref_lora_strict: bool = True,
        max_ref_tokens: int = 250,
        stt_hf_repo: str = "",
        stt_pkg_dir: str = "",
        vad_threshold: float = 0.5,
        suppress_text_during_search: bool = True,
        prompt_settle_sec: float = 0.0,
        stt_reject_foreign_script: bool = True,
        stt_max_non_latin_ratio: float = 0.15,
        stt_require_english: bool = True,
        max_input_buffer_sec: float = 2.0,
        compressor_model: str = "",
        compressor_device: str = "cuda",
        compressor_4bit: bool = True,
        compressor_max_passages: int = 2,
        router_threshold: float = 0.40,
        router_use_rules: bool = True,
        web_search_enabled: bool = False,
        web_search_api_key: str | None = None,
        web_search_provider: str = "tavily",
        web_search_max_results: int = 3,
        web_search_timeout: float = 3.0,
        web_search_min_score: float = 0.15,
        conversation_log_dir: str = "",
    ) -> None:
        from conversation_logger import ConversationLogger  # IMTalker/ is on sys.path by the time this runs

        self.conv_logger = ConversationLogger(log_dir=conversation_log_dir)

        # Seeded FIRST, before any weight loading. PEFT's LoRA init, CUDA-graph
        # capture warmup and the system-prompt replay all draw from this RNG, so
        # seeding here (rather than just before generation) makes the entire
        # startup path reproducible, not only the conversation.
        self.personaplex_seed = resolve_personaplex_seed()
        self.reseed_per_session = _env_flag("PERSONAPLEX_RESEED_PER_SESSION", True)
        if self.personaplex_seed is None:
            print(
                f"[liveTry] PersonaPlex sampling UNSEEDED "
                f"(PERSONAPLEX_SEED is empty; torch.initial_seed()={torch.initial_seed()}). "
                f"Replies will differ between launches.",
                flush=True,
            )
        else:
            seed_personaplex(self.personaplex_seed, "engine init")

        # Strict by default: every fallback below silently changes what the
        # model IS, and each one used to be invisible in the log and in the
        # notebook's health check. Set ALLOW_MOSHI_FALLBACK=1 to run degraded on
        # purpose.
        self.strict_runtime = not _env_flag("ALLOW_MOSHI_FALLBACK", False)
        self._runtime_contract: dict[str, object] = {}

        _ensure_moshi_importable(moshi_root)
        from moshi.models import LMGen, loaders

        # Recorded immediately: which moshi fork won the import is the single
        # most useful fact when comparing a good run against a bad one, and
        # nothing used to print it.
        import moshi as _moshi_pkg

        self._runtime_contract["moshi_file"] = getattr(_moshi_pkg, "__file__", "?")
        self._runtime_contract["moshi_version"] = getattr(_moshi_pkg, "__version__", "?")

        self.device = torch.device(device)
        self.placeholder_jpeg_b64 = placeholder_jpeg_b64
        self.input_buffer = np.zeros(0, dtype=np.float32)
        # Backlog cap + drop accounting (see append_browser_pcm). Set before
        # any audio can arrive so the very first append is already bounded.
        self.max_input_buffer_sec = float(max_input_buffer_sec)
        self._input_dropped_samples = 0
        # Tracked separately because they mean very different things: dropping
        # silence is free, dropping speech is a question the model never heard.
        self._input_dropped_speech_samples = 0
        self._input_drop_last_log = 0.0
        # Read by _settle_after_prompt() / _start_thinking_sound(). These MUST be
        # set before _warmup_runtime() below, which calls reset_session() ->
        # _apply_system_prompts() -> _settle_after_prompt() during __init__.
        self.prompt_settle_sec = float(prompt_settle_sec)
        self.suppress_text_during_search = bool(suppress_text_during_search)
        self.step = 0
        self.skip_first = True
        self.sampled_text = ""
        self.audio_text = ""
        self.started_at = time.perf_counter()
        self.text_prompt = str(text_prompt or "")
        self.voice_prompt = str(voice_prompt or "")
        self.voice_prompt_dir = str(voice_prompt_dir or "")
        self._hf_repo = mimi_hf_repo

        print(
            "[liveTry] loading Moshi "
            f"repo={mimi_hf_repo} root={moshi_root} device={self.device} cfg={cfg_coef}"
        )
        t0 = time.perf_counter()
        if hasattr(loaders, "CheckpointInfo"):
            ckpt_info = loaders.CheckpointInfo.from_hf_repo(mimi_hf_repo)
            self.mimi = ckpt_info.get_mimi(device=self.device)
            self.lm = ckpt_info.get_moshi(device=self.device, dtype=torch.bfloat16)
            self.tokenizer = ckpt_info.get_text_tokenizer()
            model_type = getattr(ckpt_info, "model_type", "moshi")
        else:
            from huggingface_hub import hf_hub_download
            import inspect
            import sentencepiece

            repo = mimi_hf_repo or getattr(loaders, "DEFAULT_REPO", "nvidia/personaplex-7b-v1")
            if not mimi_weight:
                mimi_weight = hf_hub_download(repo, loaders.MIMI_NAME)
            if not moshi_weight:
                moshi_weight = hf_hub_download(repo, loaders.MOSHI_NAME)
            if not tokenizer:
                tokenizer = hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME)
            self.mimi = loaders.get_mimi(mimi_weight, self.device)
            lm_kwargs = {"device": self.device, "dtype": torch.bfloat16}
            supported = set(inspect.signature(loaders.get_moshi_lm).parameters)
            optional_kwargs = {
                "quantize_4bit": bool(quantize_4bit),
                "num_codebooks": int(num_codebooks),
                "context": context,
            }
            lm_kwargs.update({k: v for k, v in optional_kwargs.items() if k in supported})
            # This signature filter is how --quantize_4bit used to vanish. Two
            # different moshi forks ship in this deployment -- the one bundled
            # inside the personaplex_bnb4 weights snapshot, and the repo's own
            # personaplex/moshi -- and only the first supports 4-bit loading. If
            # an incomplete Step 8 download made the repo copy win, the 7B model
            # was quietly loaded a different way for that whole pod.
            dropped = sorted(k for k in optional_kwargs if k not in supported)
            # Same rule as the LMGen check below: a dropped kwarg only matters
            # when its value would have changed something. `context=None` and a
            # default num_codebooks cost nothing; a dropped quantize_4bit=True
            # means the 7B model is being loaded a completely different way.
            fatal_loader = [
                k for k in dropped
                if optional_kwargs[k] not in (None, False)
                and not (k == "num_codebooks" and optional_kwargs[k] == 8)
            ]
            self._runtime_contract["dropped_loader_kwargs"] = dropped
            self._runtime_contract["fatal_loader_kwargs"] = fatal_loader
            if dropped and not fatal_loader:
                print(
                    f"[liveTry] this moshi build does not accept {dropped} -- harmless here, "
                    f"their values are defaults ("
                    + ", ".join(f"{k}={optional_kwargs[k]!r}" for k in dropped) + ")",
                    flush=True,
                )
            if fatal_loader:
                msg = (
                    f"This moshi build ignores {fatal_loader} ("
                    + ", ".join(f"{k}={optional_kwargs[k]!r}" for k in fatal_loader) + "). "
                    f"Loaded from {getattr(loaders, '__file__', '?')}, which is NOT the "
                    f"PersonaPlex fork bundled in <personaplex_bnb4>/moshi. Re-run "
                    f"scripts/download_live_assets.sh and reinstall with "
                    f"`pip install -e <personaplex_bnb4>/moshi --no-deps`, or set "
                    f"ALLOW_MOSHI_FALLBACK=1 to run degraded on purpose."
                )
                if self.strict_runtime:
                    raise RuntimeError(msg)
                print(f"[liveTry] DEGRADED: {msg}", flush=True)
            self.lm = loaders.get_moshi_lm(moshi_weight, **lm_kwargs)
            self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)  # type: ignore
            model_type = "personaplex"

        # The reference LoRA must be applied here: after the base LM is loaded,
        # before LMGen(...)/CUDA-graph capture below (and before any subclass's
        # own graph-capture warmup) -- PEFT mutates self.lm's submodules in
        # place, so the graph bakes in the LoRA-augmented forward pass either
        # way. This adapter teaches the model to consume the injected
        # <lookup>/<ref> tags; it is unrelated to where the referenced text
        # came from, so it is still required now that the text comes from a web
        # search rather than a local document index.
        self.ref_lora_dir = str(ref_lora_dir or "")
        self.ref_lora_scale = float(ref_lora_scale)
        self.ref_lora_strict = bool(ref_lora_strict)
        # True only once the adapter is really applied. `ref_lora_dir` being set
        # says only that one was REQUESTED -- the startup status line reported
        # that instead, so a failed or partial load still read as "loaded".
        self.ref_lora_active = False
        if self.ref_lora_dir:
            self._load_ref_lora(self.ref_lora_dir, merge_lora=bool(merge_ref_lora))

        self.mimi.eval()
        self.lm.eval()

        import inspect as _inspect

        _lmgen_sig = _inspect.signature(LMGen.__init__)
        _lmgen_params = set(_lmgen_sig.parameters)

        cond_tensors, cond_status = self._resolve_condition_tensors(
            model_type, float(cfg_coef), "condition_tensors" in _lmgen_params
        )
        self._runtime_contract["condition_tensors"] = len(cond_tensors or {})
        self._runtime_contract["condition_source"] = cond_status

        def on_text_hook(text_tokens: torch.Tensor) -> None:
            token = int(text_tokens[0].detach().item())
            piece = self.decode_piece(token)
            if piece:
                self.sampled_text += piece

        # Sampling temperature and top-k decide how much room the seed has to
        # move in. The TEXT stream carries the semantic content, so
        # PERSONAPLEX_TEMP_TEXT / _TOP_K_TEXT are the knobs to reach for when
        # replies drift off-topic.
        #
        # An UNSET variable passes nothing, so the build's own default applies.
        # That matters: hardcoding "0.8" here as the fallback would silently
        # change generation on any fork whose default differs, which is exactly
        # the class of accident this whole change set exists to remove. Only an
        # explicitly-set variable overrides anything.
        sampling_env = {
            "temp": ("PERSONAPLEX_TEMP", float),
            "temp_text": ("PERSONAPLEX_TEMP_TEXT", float),
            "top_k": ("PERSONAPLEX_TOP_K", int),
            "top_k_text": ("PERSONAPLEX_TOP_K_TEXT", int),
        }
        sampling_kwargs, sampling_effective, ignored_sampling = {}, {}, []
        for key, (env_name, cast) in sampling_env.items():
            param = _lmgen_sig.parameters.get(key)
            build_default = None if param is None else param.default
            raw = (os.environ.get(env_name) or "").strip()
            if not raw:
                sampling_effective[key] = f"{build_default} (build default)"
                continue
            if param is None:
                ignored_sampling.append(f"{env_name}={raw}")
                sampling_effective[key] = "<not supported by this build>"
                continue
            try:
                sampling_kwargs[key] = cast(raw)
            except ValueError:
                print(
                    f"[liveTry] {env_name}={raw!r} is not a valid {cast.__name__} "
                    f"-- ignoring it and keeping the build default {build_default!r}",
                    flush=True,
                )
                sampling_effective[key] = f"{build_default} (build default)"
                continue
            sampling_effective[key] = f"{sampling_kwargs[key]} (override)"
        if ignored_sampling:
            print(
                f"[liveTry] this LMGen build exposes no knob for {ignored_sampling} "
                f"-- those settings have no effect here",
                flush=True,
            )
        self._runtime_contract["sampling"] = sampling_effective

        # Forks differ in which of these three they expose, and a missing kwarg
        # is only a FAULT when something is actually lost. Judge each on that,
        # not on its mere absence -- an over-eager check here is just a new way
        # to fail a healthy deployment.
        core_kwargs = {
            "cfg_coef": float(cfg_coef),
            "condition_tensors": cond_tensors,
            "on_text_hook": on_text_hook,
        }
        core_missing = sorted(k for k in core_kwargs if k not in _lmgen_params)
        fatal, benign = [], []
        for k in core_missing:
            if k == "cfg_coef" and abs(float(cfg_coef) - 1.0) > 1e-6:
                # A CFG scale was explicitly requested and would be silently
                # ignored. At cfg_coef == 1.0 (the launcher's default) CFG is a
                # no-op anyway, so its absence costs nothing.
                fatal.append(k)
            elif k == "condition_tensors" and cond_tensors:
                # Tensors were actually built and could not be handed over.
                fatal.append(k)
            else:
                # on_text_hook only feeds the UI's sampled_text display; losing
                # it costs a transcript panel, not answer quality.
                benign.append(k)
        # Anything unsupported must also be dropped from the call, or the
        # construction below raises TypeError for a reason we already understand.
        for k in core_missing:
            core_kwargs.pop(k, None)
        self._runtime_contract["missing_lmgen_kwargs"] = core_missing
        self._runtime_contract["fatal_lmgen_kwargs"] = fatal
        if benign:
            print(
                f"[liveTry] this LMGen build does not accept {benign} -- harmless in this "
                f"configuration (cfg_coef={float(cfg_coef)}, "
                f"condition_tensors={len(cond_tensors or {})})",
                flush=True,
            )
        if fatal:
            msg = (
                f"This LMGen build does not accept {fatal}, and this configuration needs "
                f"them (cfg_coef={float(cfg_coef)}, "
                f"condition_tensors={len(cond_tensors or {})}) -- they would be silently "
                f"ignored. moshi loaded from "
                f"{self._runtime_contract.get('moshi_file', '?')} "
                f"(v{self._runtime_contract.get('moshi_version', '?')}). Expected the "
                f"PersonaPlex fork bundled in <personaplex_bnb4>/moshi: re-run "
                f"scripts/download_live_assets.sh and `pip install -e "
                f"<personaplex_bnb4>/moshi --no-deps`. Set ALLOW_MOSHI_FALLBACK=1 to run "
                f"degraded on purpose."
            )
            if self.strict_runtime:
                raise RuntimeError(msg)
            print(f"[liveTry] DEGRADED: {msg}", flush=True)

        # Build the call FROM the signature rather than assuming a shape. Forks
        # differ in more than which optional kwargs they take: the fork bundled
        # in the bnb4 snapshot makes `device` a REQUIRED positional, while the
        # fork this call was originally written against does not take it at all.
        # Supplying only what the signature asks for is the only version of this
        # that works across both without a hardcoded fallback.
        lmgen_kwargs = dict(core_kwargs)
        lmgen_kwargs.update(sampling_kwargs)
        if "device" in _lmgen_params:
            lmgen_kwargs["device"] = self.device

        # The model itself is passed positionally, so exclude self and the first
        # parameter; anything else left without a default is something this code
        # does not know how to supply, and guessing would be worse than saying so.
        _param_names = [n for n in _lmgen_sig.parameters if n != "self"]
        _model_param = _param_names[0] if _param_names else None
        required_unmet = [
            name
            for name, p in _lmgen_sig.parameters.items()
            if name not in ("self", _model_param)
            and p.default is _inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and name not in lmgen_kwargs
        ]
        if required_unmet:
            raise RuntimeError(
                f"This LMGen build requires {required_unmet}, which this code does not know "
                f"how to supply. Signature: LMGen{_lmgen_sig}. moshi loaded from "
                f"{self._runtime_contract.get('moshi_file', '?')}."
            )

        self.lm_gen = LMGen(self.lm, **lmgen_kwargs)
        self._runtime_contract["lmgen"] = "full" if not core_missing else "partial"
        self._runtime_contract["lmgen_kwargs"] = sorted(lmgen_kwargs)

        # One greppable line recording what this process actually is. The
        # notebook's health check keys off "PersonaPlex runtime contract OK", so
        # a degraded start can no longer report green.
        print(
            f"[liveTry] PIPELINE REVISION {PIPELINE_REVISION} "
            f"(file {Path(__file__).resolve()})",
            flush=True,
        )
        self._runtime_contract["pipeline_revision"] = PIPELINE_REVISION
        print(
            f"[liveTry] runtime contract: moshi={self._runtime_contract.get('moshi_file', '?')} "
            f"v{self._runtime_contract.get('moshi_version', '?')} "
            f"lmgen={self._runtime_contract['lmgen']} "
            f"dropped_loader_kwargs={self._runtime_contract.get('dropped_loader_kwargs', [])} "
            f"condition_tensors={self._runtime_contract['condition_tensors']}"
            f"({self._runtime_contract.get('condition_source', '?')}) "
            f"seed={self.personaplex_seed} reseed_per_session={self.reseed_per_session} "
            f"lmgen_kwargs={self._runtime_contract.get('lmgen_kwargs', [])} "
            f"sampling={sampling_effective}",
            flush=True,
        )
        # "none-declared" and "unsupported-api" are healthy: the bnb4 PersonaPlex
        # fork ships no moshi.run_inference and the model declares no
        # conditioners, so an empty dict is the correct answer, not a fault.
        contract_ok = (
            self._runtime_contract["lmgen"] != "fallback"
            and not self._runtime_contract.get("fatal_loader_kwargs")
            and not self._runtime_contract.get("fatal_lmgen_kwargs")
            and self._runtime_contract.get("condition_source") != "missing-helper"
        )
        if contract_ok:
            print("[liveTry] PersonaPlex runtime contract OK", flush=True)
        self.conv_logger.event("runtime_contract", ok=contract_ok, **{
            k: v for k, v in self._runtime_contract.items() if k != "sampling"
        })

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        if self.frame_size != FRAME_SIZE:
            raise RuntimeError(f"expected Mimi frame_size={FRAME_SIZE}, got {self.frame_size}")
        self._warmup_runtime()
        self.reset_session()
        print(f"[liveTry] Moshi ready in {time.perf_counter() - t0:.1f}s")

        # --- STT / query router / web search: all optional, each independently
        # try/except-guarded so a failure here never blocks avatar startup. ---
        self.max_ref_tokens = int(max_ref_tokens)
        self.vad_threshold = float(vad_threshold)
        self.stt_reject_foreign_script = bool(stt_reject_foreign_script)
        self.stt_max_non_latin_ratio = float(stt_max_non_latin_ratio)
        self.stt_require_english = bool(stt_require_english)
        self.web_search_enabled = bool(web_search_enabled)
        self.web_search_api_key = web_search_api_key or None
        self.web_search_provider = str(web_search_provider)
        self.web_search_max_results = int(web_search_max_results)
        self.web_search_timeout = float(web_search_timeout)
        self.web_search_min_score = float(web_search_min_score)
        if self.web_search_enabled and not self.web_search_api_key:
            print(
                "[liveTry] web_search_enabled but no web_search_api_key configured "
                "-- web search will no-op at request time",
                flush=True,
            )

        # STT no longer depends on any document index: it is the sole source of
        # the transcript the router reads, so it loads on its own merits.
        self.stt_mimi = None
        self.stt_lm_gen = None
        self.stt_tokenizer = None
        self.stt_padding_token_id = 3
        if stt_hf_repo and stt_pkg_dir:
            self._load_stt_vad(str(stt_hf_repo), str(stt_pkg_dir), self.device)

        # The compressor's small instruct model does double duty: it both
        # compresses web results into one speakable sentence AND backs the
        # query router (one shared model, one load, no extra VRAM -- see
        # QueryRouter.from_compressor). The router therefore requires the
        # compressor; if the compressor fails to load, routing is unavailable
        # and every turn is answered from the model's own knowledge.
        self.context_compressor = None
        self.query_router = None
        if self.stt_lm_gen is not None and compressor_model:
            self._load_context_compressor(
                str(compressor_model), str(compressor_device), bool(compressor_4bit), int(compressor_max_passages)
            )
            if self.context_compressor is not None:
                self._load_query_router(float(router_threshold), bool(router_use_rules))

        # True only when a transcript can be produced AND routed. When False,
        # the avatar behaves exactly like the plain conversational server: no
        # transcription side-effects, no injection, no search.
        self.search_enabled = self.stt_lm_gen is not None and self.query_router is not None
        if self.stt_lm_gen is not None and self.query_router is None:
            print(
                "[liveTry] STT loaded but no query router -- web search disabled; "
                "every turn will be answered from the model's own knowledge",
                flush=True,
            )

        # Single source of truth for "what actually came up" -- read this
        # line (also in the JSONL conversation log as kind=component_status)
        # instead of inferring readiness from scattered print statements.
        self.conv_logger.component_status(
            ref_lora_loaded=bool(self.ref_lora_active),
            ref_lora_requested=bool(self.ref_lora_dir),
            stt_loaded=self.stt_lm_gen is not None,
            compressor_loaded=self.context_compressor is not None,
            router_loaded=self.query_router is not None,
            search_enabled=self.search_enabled,
            web_search_enabled=self.web_search_enabled,
            web_search_has_key=bool(self.web_search_api_key),
        )

    def _model_conditioners(self) -> list[str]:
        """Names of the conditioners this LM actually declares, if any.

        A model with no conditioners is correctly run with an empty
        condition_tensors dict -- that is not a degraded state, and treating it
        as one turns a healthy deployment into a startup crash."""
        for attr in ("condition_provider", "conditioner_provider", "conditioners"):
            provider = getattr(self.lm, attr, None)
            if provider is None:
                continue
            conditioners = getattr(provider, "conditioners", provider)
            with contextlib.suppress(Exception):
                if isinstance(conditioners, dict):
                    return sorted(conditioners.keys())
                if hasattr(conditioners, "keys"):
                    return sorted(conditioners.keys())
        return []

    def _resolve_condition_tensors(
        self, model_type: str, cfg_coef: float, lmgen_accepts_conditions: bool
    ) -> tuple[dict, str]:
        """Build the CFG condition tensors, or establish that this build has
        none to build.

        `moshi.run_inference` is a convenience module that not every PersonaPlex
        fork ships -- the one bundled in the bnb4 weights snapshot does not, and
        that fork is the CORRECT one. So a missing import here says nothing on
        its own; what matters is whether the loaded model declares conditioners
        that are then going unused. Only that last case is a real fault.

        Returns (tensors, status) where status is one of:
          built            -- tensors were produced by a resolved helper
          none-declared    -- the model has no conditioners; {} is correct
          unsupported-api  -- this LMGen takes no condition_tensors at all
          missing-helper   -- conditioners exist but no helper could build them
        """
        candidates = (
            "moshi.run_inference",
            "moshi.models.loaders",
            "moshi.models.lm",
            "moshi.models",
            "moshi.conditioners",
            "moshi.utils",
        )
        getter = None
        for module_name in candidates:
            with contextlib.suppress(Exception):
                module = __import__(module_name, fromlist=["get_condition_tensors"])
                getter = getattr(module, "get_condition_tensors", None)
                if getter is not None:
                    break

        if getter is not None:
            try:
                tensors = getter(model_type, self.lm, batch_size=1, cfg_coef=cfg_coef)
                print(
                    f"[liveTry] condition tensors: {len(tensors or {})} built by "
                    f"{getattr(getter, '__module__', '?')}.get_condition_tensors "
                    f"(cfg_coef={cfg_coef})",
                    flush=True,
                )
                return tensors or {}, "built"
            except Exception as e:
                tb = traceback.format_exc()
                msg = (
                    f"get_condition_tensors was found but failed ({e!r}); PersonaPlex would run "
                    f"UNCONDITIONED. Set ALLOW_MOSHI_FALLBACK=1 to run anyway."
                )
                if self.strict_runtime:
                    raise RuntimeError(msg) from e
                print(f"[liveTry] DEGRADED: {msg}\n{tb}", flush=True)
                self.conv_logger.error("condition_tensors", e, tb)
                return {}, "missing-helper"

        if not lmgen_accepts_conditions:
            print(
                "[liveTry] condition tensors: not applicable -- this LMGen build takes no "
                "condition_tensors argument",
                flush=True,
            )
            return {}, "unsupported-api"

        conditioners = self._model_conditioners()
        if not conditioners:
            # The common, healthy case for the bnb4 fork: no run_inference
            # module and no conditioners on the model, so {} is exactly right.
            print(
                "[liveTry] condition tensors: none -- this model declares no conditioners "
                "(no moshi.run_inference in this build, and none needed)",
                flush=True,
            )
            return {}, "none-declared"

        msg = (
            f"This model declares conditioners {conditioners} but no get_condition_tensors "
            f"helper could be resolved (tried {list(candidates)}), so PersonaPlex would run "
            f"UNCONDITIONED. moshi loaded from "
            f"{self._runtime_contract.get('moshi_file', '?')}. Set ALLOW_MOSHI_FALLBACK=1 to "
            f"run without conditioning on purpose."
        )
        if self.strict_runtime:
            raise RuntimeError(msg)
        print(f"[liveTry] DEGRADED: {msg}", flush=True)
        return {}, "missing-helper"

    def _load_ref_lora(self, checkpoint_dir: str, merge_lora: bool = False) -> None:
        """Load the <lookup>/<ref> LoRA adapter onto self.lm. Unmerged by
        default (QLoRA-style: LoRA computed at forward time on top of the 4-bit
        base, not merged into the quantized weights). PEFT mutates self.lm's
        target submodules in place -- self.lm keeps pointing at the same, now
        LoRA-augmented, object either way."""
        lora_path = Path(checkpoint_dir) / "lora"
        if not lora_path.exists():
            print(f"[liveTry] no lora/ at {lora_path} -- skipping", flush=True)
            return
        try:
            from peft import PeftModel

            print(f"[liveTry] loading reference LoRA from {lora_path} (merge={merge_lora})", flush=True)
            peft_model = PeftModel.from_pretrained(self.lm, str(lora_path))
            if merge_lora:
                self.lm = peft_model.merge_and_unload()
            print(f"[liveTry] reference LoRA loaded from {lora_path}", flush=True)
            self._apply_ref_lora_scale(peft_model)
            # Deliberately OUTSIDE the except below: a coverage failure under
            # REF_LORA_STRICT must stop startup, not be caught and downgraded to
            # "continuing without it" like a genuine load error.
            self.ref_lora_active = True
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[liveTry] reference LoRA load failed (continuing without it): {e!r}\n{tb}", flush=True)
            self.conv_logger.error("ref_lora_load", e, tb)
            self.ref_lora_active = False
            self._peft_model = None
            return

        # Coverage is judged here, outside the try, so REF_LORA_STRICT can stop
        # the server. A half-applied adapter is the failure that looks healthy.
        self._report_ref_lora_coverage(peft_model, lora_path)

    @torch.no_grad()
    def _apply_ref_lora_scale(self, peft_model) -> None:
        """Dial the reference LoRA's strength up or down.

        The adapter ships r=128 / alpha=256, i.e. an effective scaling of 2.0
        over the 4-bit base. That is strong enough to bring its training
        corpus's persona along with the <lookup>/<ref> syntax it was meant to
        teach -- conversation_logs_1 shows the assistant answering "What is
        Bitcoin?" as though it were support for an app with a virtual currency.
        Scaling below 1.0 keeps the tag handling while weakening that pull;
        REF_LORA_SCALE=0 makes it a no-op without changing the load path.
        """
        scale = float(getattr(self, "ref_lora_scale", 1.0))
        if abs(scale - 1.0) < 1e-9:
            return
        try:
            touched = 0
            for module in peft_model.modules():
                scaling = getattr(module, "scaling", None)
                if isinstance(scaling, dict) and scaling:
                    for adapter_name in list(scaling):
                        scaling[adapter_name] = scaling[adapter_name] * scale
                        touched += 1
            print(
                f"[liveTry] reference LoRA scaled by {scale} across {touched} adapter "
                f"layer(s) -- lower values keep the <ref> syntax while weakening the "
                f"adapter's own persona",
                flush=True,
            )
            self.conv_logger.event("ref_lora_scaled", scale=scale, layers=touched)
        except Exception as e:
            print(f"[liveTry] reference LoRA scaling failed (running at full strength): {e!r}",
                  flush=True)

    @torch.no_grad()
    def _report_ref_lora_coverage(self, peft_model, lora_path) -> None:
        """Say how much of the reference LoRA is actually doing anything.

        PEFT wraps every module matching adapter_config.json's `target_modules`,
        which for this hand-written config is a broad substring list
        ("proj", "linear", "fc1", ...). Modules the checkpoint has no weights for
        are still wrapped, then default-initialised with lora_A random and
        lora_B ZERO -- so their delta is exactly zero and they are no-ops. That
        is what PEFT's "missing adapter keys" warning about
        depformer.*.self_attn.in_proj means: harmless, but it also means the
        adapter covers less of the model than the config implies, and nothing
        used to say by how much.
        """
        try:
            wrapped: set[str] = set()
            zero_b: set[str] = set()
            for name, module in peft_model.named_modules():
                lora_B = getattr(module, "lora_B", None)
                if lora_B is None or not hasattr(lora_B, "keys"):
                    continue
                # Normalise into the namespace the checkpoint keys use, so the
                # two sets are directly comparable.
                short = name
                for prefix in ("base_model.model.", "base_model."):
                    if short.startswith(prefix):
                        short = short[len(prefix):]
                        break
                wrapped.add(short)
                for adapter_name in list(lora_B.keys()):
                    weight = getattr(lora_B[adapter_name], "weight", None)
                    if weight is not None and float(weight.detach().abs().max().item()) == 0.0:
                        zero_b.add(short)
            if not wrapped:
                return
        except Exception as e:
            # Measuring must never break startup.
            print(f"[liveTry] reference LoRA coverage report failed (ignored): {e!r}", flush=True)
            return

        # Judge the load by NAMES, not by values.
        #
        # Two earlier versions of this check got that wrong and refused to start
        # a correctly-loaded adapter. The second one used "lora_B is all zeros"
        # as a proxy for "this module never received its weights" -- but a zero
        # lora_B is a perfectly legitimate thing for a checkpoint to CONTAIN. It
        # means that module contributes nothing to the forward pass, which is a
        # fact about how the adapter was trained, not about whether it loaded.
        # This deployment's checkpoint carries exactly that: 38 modules, 6 of
        # them (depformer.*.self_attn.out_proj) with a zero B.
        #
        # The real question is whether every module the checkpoint names got
        # wrapped, so its weights had somewhere to land. That is a set
        # comparison, and it cannot be confused by the weights' values.
        ckpt = _checkpoint_lora_modules(Path(lora_path) / "adapter_model.safetensors")
        if ckpt is None:
            print(
                f"[liveTry] reference LoRA: {len(wrapped)} module(s) wrapped; the checkpoint "
                f"could not be read, so coverage is unverified (starting anyway)",
                flush=True,
            )
            return

        missing = sorted(ckpt - wrapped)
        extra = sorted(wrapped - ckpt)
        print(
            f"[liveTry] reference LoRA coverage: checkpoint names {len(ckpt)} module(s), "
            f"PEFT wrapped {len(wrapped)}, missing {len(missing)}, extra {len(extra)}, "
            f"zero-valued lora_B {len(zero_b)}",
            flush=True,
        )
        self.conv_logger.event(
            "ref_lora_coverage", path=str(lora_path),
            checkpoint_modules=len(ckpt), wrapped_modules=len(wrapped),
            missing_modules=len(missing), extra_modules=len(extra),
            zero_valued_lora_b=len(zero_b),
        )

        if missing:
            msg = (
                f"reference LoRA is only PARTIALLY applied: the checkpoint carries weights for "
                f"{len(ckpt)} module(s), but {len(missing)} of them were never wrapped, so their "
                f"weights had nowhere to load. First few: {missing[:5]}. "
                f"adapter_config.json does not match "
                f"{lora_path}/adapter_model.safetensors. Regenerate it:\n"
                f"    python stability/derive_ref_lora_config.py {lora_path}\n"
                f"Set REF_LORA_STRICT=0 to start anyway."
            )
            if getattr(self, "ref_lora_strict", False):
                raise RuntimeError(msg)
            print(f"[liveTry] WARNING {msg}", flush=True)
            return

        print(
            f"[liveTry] reference LoRA fully applied: every one of the {len(ckpt)} module(s) in "
            f"the checkpoint is wrapped and loaded -- injected reference blocks will be handled "
            f"as trained",
            flush=True,
        )
        if extra:
            print(
                f"[liveTry] ({len(extra)} module(s) were wrapped that the checkpoint does not "
                f"name; PEFT gives those lora_B = 0, so they are exact no-ops. Harmless.)",
                flush=True,
            )
        if zero_b:
            print(
                f"[liveTry] (note: {len(zero_b)} module(s) have an all-zero lora_B and therefore "
                f"contribute nothing to the forward pass. That is how the checkpoint was saved, "
                f"not a load failure. e.g. {sorted(zero_b)[:3]})",
                flush=True,
            )

    def _load_stt_vad(self, stt_hf_repo: str, stt_pkg_dir: str, device) -> None:
        # Import search_helpers first, on its own, with a specific error
        # message -- a missing IMTalker/search_helpers.py on the deployed
        # checkout is the single most likely cause of a silent "search
        # disabled" (it's a plain ModuleNotFoundError that the broad except
        # below would otherwise bury under a generic message).
        try:
            import search_helpers
        except ImportError as e:
            tb = traceback.format_exc()
            print(
                f"[liveTry] search disabled -- could not import search_helpers ({e!r}). "
                f"Check that IMTalker/search_helpers.py exists in this checkout (a missing file "
                f"here is the most common cause of search silently not loading).\n{tb}",
                flush=True,
            )
            self.conv_logger.error("search_helpers_import", e, tb)
            return

        try:
            print(f"[liveTry] loading STT model from {stt_hf_repo}...", flush=True)
            moshi_stt = search_helpers.load_upstream_moshi_stt(stt_pkg_dir)
            stt_info = moshi_stt.models.loaders.CheckpointInfo.from_hf_repo(stt_hf_repo)
            self.stt_mimi = stt_info.get_mimi(device=device)
            stt_lm = stt_info.get_moshi(device=device, dtype=torch.bfloat16)
            stt_lm.eval()
            self.stt_lm_gen = moshi_stt.models.LMGen(stt_lm, temp=0, temp_text=0.0)
            self.stt_lm_gen.streaming_forever(1)
            self.stt_mimi.streaming_forever(1)
            self.stt_tokenizer = stt_info.get_text_tokenizer()
            self.stt_padding_token_id = stt_info.raw_config.get("text_padding_token_id", 3)
            # Print the identity of BOTH tokenizers. They must be different
            # objects with different vocabularies: the STT one decodes the STT
            # model's text stream, PersonaPlex's 32k multilingual SPM encodes
            # the injected <lookup>/<ref> tags. If a transcript ever comes out
            # in a script the en/fr STT model cannot produce, compare these two
            # numbers first -- matching vocab sizes here would mean the STT
            # stream is being decoded by the wrong vocabulary, which produces
            # exactly that symptom (real token ids, plausible words, wrong
            # language).
            stt_desc = search_helpers.describe_tokenizer(self.stt_tokenizer)
            main_desc = search_helpers.describe_tokenizer(self.tokenizer)
            print(
                f"[liveTry] STT model loaded: "
                f"params={sum(p.numel() for p in stt_lm.parameters()) / 1e9:.1f}B "
                f"padding_id={self.stt_padding_token_id}",
                flush=True,
            )
            print(f"[liveTry] stt tokenizer={stt_desc}  |  main tokenizer={main_desc}", flush=True)
            if stt_desc == main_desc:
                print(
                    "[liveTry] WARNING the STT and PersonaPlex tokenizers look identical. "
                    "The STT text stream may be decoded with the wrong vocabulary, which shows up "
                    "as transcripts in a language the en/fr STT model cannot actually produce.",
                    flush=True,
                )
            self.conv_logger.event(
                "stt_loaded", stt_repo=stt_hf_repo, stt_tokenizer=stt_desc,
                main_tokenizer=main_desc, padding_token_id=self.stt_padding_token_id,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[liveTry] search disabled -- STT model load failed: {e!r}\n{tb}", flush=True)
            self.conv_logger.error("stt_load", e, tb)
            self.stt_mimi = None
            self.stt_lm_gen = None

    def _load_context_compressor(
        self, compressor_model: str, compressor_device: str, compressor_4bit: bool, compressor_max_passages: int
    ) -> None:
        try:
            import search_helpers

            self.context_compressor = search_helpers.ContextCompressor(
                model_name=compressor_model,
                device=compressor_device,
                quantize_4bit=compressor_4bit,
                max_passages=compressor_max_passages,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(
                f"[liveTry] context compressor disabled -- load failed: {e!r} "
                f"(web search will be disabled too: the router shares this model)\n{tb}",
                flush=True,
            )
            self.conv_logger.error("compressor_load", e, tb)
            self.context_compressor = None

    def _load_query_router(self, threshold: float, use_rules: bool) -> None:
        """Build the search/no-search router on top of the already-loaded
        compressor model. Shares weights, so this costs no extra VRAM and no
        extra load time -- only the Yes/No token-id lookup."""
        try:
            import search_helpers

            self.query_router = search_helpers.QueryRouter.from_compressor(
                self.context_compressor, threshold=threshold, use_rules=use_rules,
            )
            print(
                f"[liveTry] query router ready (threshold={threshold:.2f} "
                f"rules={'on' if use_rules else 'off'}, sharing the compressor model)",
                flush=True,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(
                f"[liveTry] query router disabled -- init failed: {e!r} "
                f"(every turn will be answered from the model's own knowledge)\n{tb}",
                flush=True,
            )
            self.conv_logger.error("router_load", e, tb)
            self.query_router = None

    def _resolve_voice_prompt_path(self) -> str:
        if not self.voice_prompt:
            return ""
        if os.path.isabs(self.voice_prompt) and os.path.exists(self.voice_prompt):
            return self.voice_prompt
        if self.voice_prompt_dir and os.path.isdir(self.voice_prompt_dir):
            candidate = os.path.join(self.voice_prompt_dir, self.voice_prompt)
            if os.path.exists(candidate):
                return candidate
        from huggingface_hub import hf_hub_download

        voices_tgz = Path(hf_hub_download(self._hf_repo, "voices.tgz"))
        voices_dir = voices_tgz.parent / "voices"
        if not voices_dir.exists():
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_tgz.parent)
        candidate = voices_dir / self.voice_prompt
        if not candidate.exists():
            raise FileNotFoundError(f"voice prompt not found: {candidate}")
        self.voice_prompt_dir = str(voices_dir)
        return str(candidate)

    @torch.no_grad()
    def _apply_system_prompts(self) -> None:
        if not hasattr(self.lm_gen, "step_system_prompts"):
            return
        voice_path = self._resolve_voice_prompt_path()
        raw_voice_prompt = bool(voice_path and not voice_path.endswith(".pt"))
        if voice_path:
            if voice_path.endswith(".pt") and hasattr(self.lm_gen, "load_voice_prompt_embeddings"):
                self.lm_gen.load_voice_prompt_embeddings(voice_path)
            elif hasattr(self.lm_gen, "load_voice_prompt"):
                self.lm_gen.load_voice_prompt(voice_path)
            print(f"[liveTry] voice prompt: {voice_path}", flush=True)
            # Report what the voice prompt carries, for startup visibility.
            #
            # NOTE, corrected: `voice_prompt_cache` is NOT a saved conversation.
            # _init_streaming_state allocates state.cache as
            #     (batch, num_codebooks, max_delay + 3)
            # i.e. roughly five to eleven slots -- a codebook-DELAY alignment
            # ring buffer, not context. `state.cache.copy_(voice_prompt_cache)`
            # in _step_voice_prompt_core simply restores that delay alignment so
            # generation continues cleanly after the prompt. Conversational
            # context lives in the transformer KV state, not here.
            with contextlib.suppress(Exception):
                emb = getattr(self.lm_gen, "voice_prompt_embeddings", None)
                cache = getattr(self.lm_gen, "voice_prompt_cache", None)
                if cache is not None:
                    n_emb = int(emb.shape[0]) if emb is not None else 0
                    print(
                        f"[liveTry] voice prompt: {n_emb} embedding frames, "
                        f"delay-alignment cache {tuple(cache.shape)}",
                        flush=True,
                    )
                    self.conv_logger.event(
                        "voice_prompt_loaded", path=str(voice_path),
                        cache_shape=list(cache.shape), n_embedding_frames=n_emb,
                    )

        if self.text_prompt and hasattr(self.tokenizer, "encode"):
            with contextlib.suppress(Exception):
                # The system prompt is NOT a separate role for this model: the
                # fork's _step_text_prompt_core() force-feeds these ids through
                # lm_gen.step(text_token=...), i.e. the same path used to inject
                # <lookup>/<ref>. From the model's point of view it just SAID
                # the prompt out loud.
                #
                # The <|im_start|>/<|im_end|> wrapper is ChatML, a Qwen
                # convention. It only helps if this SentencePiece vocabulary
                # contains those markers as single tokens. It does not --
                # conversation_log_3 measured marker_token_count=7 -- so
                # wrapping would spell `< | im _ start | > system` out to the
                # model before the instructions. Measure, then choose.
                marker_ids = self.tokenizer.encode("<|im_start|>")
                chatml_supported = 0 < len(marker_ids) <= 2
                if chatml_supported:
                    wrapped = f"<|im_start|>system\n{self.text_prompt}<|im_end|>\n"
                else:
                    wrapped = self.text_prompt
                    print(
                        f"[liveTry] tokenizer has no ChatML markers "
                        f"('<|im_start|>' -> {len(marker_ids)} tokens); feeding the system prompt "
                        f"as plain text instead of spelling the markers out to the model",
                        flush=True,
                    )
                self.lm_gen.text_prompt_tokens = self.tokenizer.encode(wrapped)
                print(
                    f"[liveTry] text prompt loaded: {len(self.lm_gen.text_prompt_tokens)} tokens, "
                    f"chatml={chatml_supported}: {self.text_prompt[:80]!r}",
                    flush=True,
                )
                self.conv_logger.event(
                    "text_prompt_loaded",
                    chatml_markers_supported=chatml_supported,
                    marker_token_count=len(marker_ids),
                    n_prompt_tokens=len(self.lm_gen.text_prompt_tokens),
                )

        encoder_graph = None
        if raw_voice_prompt:
            state = getattr(self.mimi, "_streaming_state", None)
            encoder_graph = getattr(state, "graphed_tr_enc", None)
            if encoder_graph is not None:
                encoder_graph.disable = True
        try:
            # THIS is what actually applies both prompts: it replays the voice
            # prompt through the model (setting the speaker timbre) and then
            # feeds the system prompt. Loading the voice prompt above only
            # populates lm_gen.voice_prompt_embeddings -- without this call the
            # model never hears VARM3 and speaks in its default voice.
            self.lm_gen.step_system_prompts(self.mimi)
            self._settle_after_prompt()
        finally:
            with contextlib.suppress(Exception):
                self.mimi.reset_streaming()
            if encoder_graph is not None:
                encoder_graph.reset()
                encoder_graph.disable = False

    @torch.no_grad()
    def _settle_after_prompt(self) -> None:
        """Force a stretch of 'the assistant said nothing' after the system
        prompt.

        Because the prompt arrives as the model's own speech (see
        _apply_system_prompts), generation resumes from a context that ends
        mid-self-description, and the model's most natural continuation is
        MORE self-description. conversation_log_2 shows the result: the reply
        to the first real question was "with basic banking, investment, and
        direct financial questions. I also have some knowledge about enterprise
        growth, real estate, and AI-related topics." -- the tail of the
        prompt's capability list, not an answer.

        The fork already appends ~0.5s of silence (audio_silence_frame_cnt),
        which is not enough to read as end-of-turn. These extra padded steps
        use the model's own zero_text_code, exactly as _step_audio_silence_core
        does, so the context ends with a clear silent gap instead.

        conversation_logs_4 is the same failure again, and shows how far it
        reaches. The prompt ends "...promote RB Labs robots when relevant", so
        the model -- which believes it just SAID that -- answered "What is
        Bitcoin?" with "I'd say it's because they're a small team with a big
        vision. They've been working on AI stuff for years". Because the KV
        cache is never reset per turn, that monologue then ran the whole
        session: it swallowed the next question, and it swallowed two correctly
        retrieved <ref> facts as well. In conversation_logs_3 the same build
        looked fine only because the first question there was "What is your
        name?", which self-description happens to answer.

        The user stream carries SILENCE here, not a sine tone. That is the whole
        point of the step. A sine tone on the user stream is this fork's marker
        for the priming region (see _inject_tokens in the AH server for the
        evidence), so padding with sine kept the model INSIDE priming and simply
        made it quieter -- which is why an unbounded version of this once
        produced a mute session. Silence on the user stream is an ordinary
        conversational gap, so the model exits priming and arrives at a turn
        boundary, waiting for the user.

        Length defaults to the same 0.96s this system already treats as
        end-of-utterance (_VAD_SILENCE_FRAMES_REQUIRED). Set
        PROMPT_SETTLE_USER_STREAM=sine to restore the old register for A/B."""
        n = int(round(float(self.prompt_settle_sec) * TARGET_SR / FRAME_SIZE))
        if n <= 0:
            print(
                "[liveTry] prompt settle DISABLED (--prompt_settle_sec 0): the model resumes "
                "immediately after the system prompt, which it heard as its own speech, and may "
                "carry on describing itself instead of answering the first question",
                flush=True,
            )
            return
        zero_text = getattr(self.lm_gen, "zero_text_code", 3)
        use_sine = os.environ.get("PROMPT_SETTLE_USER_STREAM", "silence").strip().lower() == "sine"
        user_frame = (
            self.lm_gen._encode_sine_frame() if use_sine else self.lm_gen._encode_zero_frame()
        )
        try:
            for _ in range(n):
                self.lm_gen.step(
                    moshi_tokens=self.lm_gen._encode_zero_frame(),
                    text_token=zero_text,
                    input_tokens=user_frame,
                )
            print(
                f"[liveTry] prompt settle: {n} frames ({self.prompt_settle_sec:.2f}s) of "
                f"{'priming' if use_sine else 'conversational'} silence after the system prompt, "
                f"so the model reaches a turn boundary instead of carrying on describing itself",
                flush=True,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[liveTry] prompt settle failed (continuing): {e!r}\n{tb}", flush=True)
            self.conv_logger.error("prompt_settle", e, tb)

    def reset_session(self) -> None:
        # Re-seed here, not only at engine init. reset_session runs once per
        # WebSocket connect and re-applies the voice + system prompts, and the
        # tail of that replay is sampled -- so without this, every browser
        # reconnect started the conversation from a different hidden state even
        # within one server process. Re-seeding makes connection N identical to
        # connection 1, which is what makes an A/B comparison meaningful.
        if getattr(self, "reseed_per_session", False):
            seed_personaplex(getattr(self, "personaplex_seed", None), "session reset")
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.step = 0
        self.skip_first = True
        self.sampled_text = ""
        self.audio_text = ""
        self.started_at = time.perf_counter()
        with contextlib.suppress(Exception):
            self.mimi.reset_streaming()
        with contextlib.suppress(Exception):
            self.lm_gen.reset_streaming()
        # Guarded with getattr: reset_session() is also called from inside
        # _warmup_runtime(), before the STT submodel (loaded at the very end
        # of __init__, after warmup) exists yet.
        stt_lm_gen = getattr(self, "stt_lm_gen", None)
        if stt_lm_gen is not None:
            with contextlib.suppress(Exception):
                stt_lm_gen.reset_streaming()
            with contextlib.suppress(Exception):
                self.stt_mimi.reset_streaming()
        self._apply_system_prompts()

    @torch.no_grad()
    def _warmup_runtime(self, n_steps: int = 6) -> None:
        t0 = time.perf_counter()
        silence = torch.zeros(1, 1, self.frame_size, device=self.device, dtype=torch.float32)
        for idx in range(int(n_steps)):
            codes = self.mimi.encode(silence)
            if idx == 0:
                self.mimi.reset_streaming()
            tokens = self.lm_gen.step(codes[:, :, :1])
            if tokens is not None:
                reply = self.mimi.decode(tokens[:, 1:])
                _ = reply.detach().float().mean().item()
        self.reset_session()
        _sync = getattr(torch.cuda, "synchronize", None)
        if callable(_sync) and torch.cuda.is_available():
            _sync()
        print(f"[liveTry] Moshi runtime warmup done in {1000.0 * (time.perf_counter() - t0):.0f}ms")

    def decode_piece(self, token: int) -> str:
        if token in (0, 3):
            return ""
        with contextlib.suppress(Exception):
            return _clean_text_piece(self.tokenizer.id_to_piece(int(token)))
        return ""

    def append_browser_pcm(self, pcm_i16: np.ndarray, input_sr: int) -> None:
        pcm = pcm_i16.astype(np.float32) / 32768.0
        if int(input_sr) != TARGET_SR:
            wav = torch.from_numpy(pcm).view(1, -1)
            pcm = torchaudio.functional.resample(wav, int(input_sr), TARGET_SR)[0].numpy()
        self.input_buffer = np.concatenate([self.input_buffer, pcm.astype(np.float32, copy=False)])

        # -- Bound the mic backlog -------------------------------------------
        # This buffer used to be unbounded, and that made any backlog PERMANENT
        # rather than temporary. The GPU producer is rate-limited to exactly
        # real time by frame_q backpressure (it blocks once ~32 rendered frames
        # are queued, and the sender drains those at 25fps), so it consumes
        # 0.96s of microphone audio per 0.96s of wall clock and can never run
        # fast enough to catch up. Whatever backlog accumulates -- during model
        # warmup, system-prompt stepping, or any transient stall -- therefore
        # becomes a fixed end-to-end delay that persists for the whole session.
        #
        # Confirmed in conversation_log_1: component_status showed 50 avatar
        # chunks produced in 47.99s and again in 48.03s (0.96s/chunk, exactly
        # real time, never faster), while the assistant's replies were landing
        # tens of seconds after the question that prompted them.
        #
        # Dropping the OLDEST audio keeps the newest, which is what a live
        # conversation needs: being 30s behind is far worse than missing the
        # first moments of a sentence. Every drop is logged, because silently
        # discarding user speech would be its own bug.
        # -- Drop from SILENCE, never from the middle of a word ---------------
        # The trim above the cap used to slice the oldest samples unconditionally.
        # Because the GPU producer is pinned to real time by frame_q
        # backpressure, the backlog sits permanently near the cap (measured at
        # 1.24-1.44s in conversation_logs_2 with a 2.0s cap), so those slices
        # landed inside whatever the user happened to be saying. That session
        # discarded 4.5s of speech, most of it in the first minute, and the
        # damage is visible in the replies: the model answered "What is
        # Bitcoin?" with "A beat is a measure of timing" -- it never heard the
        # whole word -- and on the very first turn said outright "Hmm, I can't
        # hear you well. Can you speak up or move closer?"
        #
        # Trimming a silent gap costs nothing: it removes latency without
        # removing information. So look for silence in the oldest audio first,
        # and only cut into speech once the buffer passes a hard ceiling, where
        # unbounded delay is the worse failure.
        max_samples = int(float(getattr(self, "max_input_buffer_sec", 0.0)) * TARGET_SR)
        if max_samples > 0 and self.input_buffer.shape[0] > max_samples:
            want = int(self.input_buffer.shape[0] - max_samples)
            excess = self.input_buffer[:want]
            # Scan the oldest `want` samples in 20ms steps and keep only the
            # leading run that is quiet enough to be a gap between words.
            step = max(1, TARGET_SR // 50)
            quiet = 0
            while quiet + step <= excess.shape[0]:
                window = excess[quiet:quiet + step]
                if float(np.sqrt(np.mean(np.square(window, dtype=np.float32)))) > self._INPUT_SILENCE_RMS:
                    break
                quiet += step
            hard_ceiling = int(
                float(getattr(self, "max_input_buffer_sec", 0.0))
                * self._INPUT_HARD_CEILING_FACTOR * TARGET_SR
            )
            if quiet >= want:
                dropped = want                      # the whole excess was silence
            elif self.input_buffer.shape[0] > hard_ceiling:
                dropped = want                      # too far behind; latency now wins
            elif quiet > 0:
                dropped = quiet                     # trim only the silent lead-in
            else:
                return                              # all speech, still under the ceiling: keep it
            self.input_buffer = self.input_buffer[dropped:].copy()
            self._input_dropped_samples += dropped
            self._input_dropped_speech_samples += max(0, dropped - quiet)
            now = time.perf_counter()
            if now - self._input_drop_last_log >= 2.0:
                self._input_drop_last_log = now
                total_s = self._input_dropped_samples / TARGET_SR
                speech_s = self._input_dropped_speech_samples / TARGET_SR
                kind = "silence" if dropped == quiet else "SPEECH"
                print(
                    f"[liveTry] microphone backlog exceeded {max_samples / TARGET_SR:.2f}s -- "
                    f"trimmed {dropped / TARGET_SR:.2f}s of {kind} "
                    f"(session total {total_s:.1f}s, of which {speech_s:.1f}s was speech). "
                    + (
                        "Trimming silence is free. "
                        if speech_s <= 0.0 else
                        "SPEECH WAS DISCARDED: the model did not hear all of the question. "
                        "Lower the render cost (--render_sub_batch / --jpeg_quality / --nfe) "
                        "or raise --max_input_buffer_sec. "
                    ),
                    flush=True,
                )
                self.conv_logger.event(
                    "input_backlog_drop",
                    f"dropped={dropped / TARGET_SR:.2f}s of {kind} total={total_s:.1f}s "
                    f"speech={speech_s:.1f}s",
                    dropped_s=round(dropped / TARGET_SR, 3),
                    dropped_kind=kind,
                    total_dropped_s=round(total_s, 2),
                    total_dropped_speech_s=round(speech_s, 2),
                    cap_s=round(max_samples / TARGET_SR, 2),
                )

    def input_backlog_sec(self) -> float:
        """Seconds of microphone audio waiting to be processed. This is the
        single best predictor of how late the next reply will be: the producer
        runs at real time, so a backlog here is a delay that will not shrink on
        its own."""
        return float(self.input_buffer.shape[0]) / TARGET_SR

    @torch.no_grad()
    def process_ready_steps(self) -> list[dict]:
        events: list[dict] = []
        while self.input_buffer.shape[0] >= FRAME_SIZE:
            pcm = self.input_buffer[:FRAME_SIZE].copy()
            self.input_buffer = self.input_buffer[FRAME_SIZE:].copy()
            events.append(self._step(pcm))
        return events

    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:
        self.step += 1
        t0 = time.perf_counter()
        chunk = torch.from_numpy(pcm24).to(self.device, dtype=torch.float32)[None, None]

        t_encode0 = time.perf_counter()
        codes = self.mimi.encode(chunk)
        t_encode1 = time.perf_counter()
        if self.skip_first:
            # Same first-frame reset used in Moshi examples/live code.
            self.mimi.reset_streaming()
            self.skip_first = False

        t_lm0 = time.perf_counter()
        tokens = self.lm_gen.step(codes[:, :, :1])
        t_lm1 = time.perf_counter()

        token = -1
        token_piece = ""
        decode_ms = 0.0
        if tokens is None:
            reply_pcm = np.zeros(FRAME_SIZE, dtype=np.float32)
        else:
            token = int(tokens[0, 0, 0].detach().item())
            token_piece = self.decode_piece(token)
            if token_piece:
                self.audio_text += token_piece
            t_decode0 = time.perf_counter()
            reply = self.mimi.decode(tokens[:, 1:])
            reply_pcm = reply[0, 0].detach().float().cpu().numpy()
            decode_ms = 1000.0 * (time.perf_counter() - t_decode0)
            if reply_pcm.shape[0] < FRAME_SIZE:
                reply_pcm = np.pad(reply_pcm, (0, FRAME_SIZE - reply_pcm.shape[0]))
            elif reply_pcm.shape[0] > FRAME_SIZE:
                reply_pcm = reply_pcm[:FRAME_SIZE]

        reply_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))
        reply_peak = float(np.max(np.abs(reply_pcm))) if reply_pcm.size else 0.0
        input_rms = float(np.sqrt(np.mean(np.square(pcm24, dtype=np.float32))))
        encode_ms = 1000.0 * (t_encode1 - t_encode0)
        lm_ms = 1000.0 * (t_lm1 - t_lm0)
        total_ms = 1000.0 * (time.perf_counter() - t0)

        reply_i16 = np.clip(reply_pcm, -1.0, 1.0)
        reply_i16 = (reply_i16 * 32767.0).astype(np.int16)
        audio_b64 = base64.b64encode(reply_i16.tobytes()).decode("ascii")

        print(
            "[liveTry] moshi "
            f"step={self.step} token={token} piece={token_piece!r} "
            f"in_rms={input_rms:.5f} reply_rms={reply_rms:.5f} peak={reply_peak:.3f} "
            f"encode={encode_ms:.1f}ms lm={lm_ms:.1f}ms decode={decode_ms:.1f}ms total={total_ms:.1f}ms"
        )

        return {
            "step": int(self.step),
            "sample_rate": TARGET_SR,
            "reply_i16_b64": audio_b64,
            "reply_rms": reply_rms,
            "reply_peak": reply_peak,
            "input_rms": input_rms,
            "token": token,
            "piece": token_piece,
            "sampled_text": self.sampled_text,
            "audio_text": self.audio_text,
            "encode_ms": encode_ms,
            "lm_ms": lm_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
        }


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="IMTalker Moshi liveTry")
    started_at = time.perf_counter()
    html_path = Path(args.html_path)
    placeholder_jpeg_b64 = _make_placeholder_jpeg(args.placeholder_path)
    engine: MoshiOnlyEngine | None = None

    def get_engine() -> MoshiOnlyEngine:
        nonlocal engine
        if engine is None:
            engine = MoshiOnlyEngine(
                moshi_root=args.moshi_root,
                mimi_hf_repo=args.mimi_hf_repo,
                device=args.device,
                cfg_coef=args.cfg_coef,
                placeholder_jpeg_b64=placeholder_jpeg_b64,
            )
        return engine

    @app.get("/")
    async def index():
        if html_path.is_file():
            return FileResponse(
                html_path,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return HTMLResponse(
            f"<h1>Missing HTML</h1><p>Expected: {html_path}</p>",
            status_code=500,
        )

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "stage": "moshi_text_audio_only",
            "uptime_sec": round(time.perf_counter() - started_at, 3),
            "moshi_loaded": engine is not None,
        })

    @app.websocket("/ws/conversation")
    async def conversation(ws: WebSocket):
        await ws.accept()
        input_sr = 48000
        packets = 0
        samples = 0
        t0 = time.perf_counter()
        moshi = get_engine()

        await ws.send_json({
            "type": "server_ready",
            "sample_rate": TARGET_SR,
            "model_type": "moshi-only",
            "tokens_per_chunk": 1,
            "buffer_ms": 400,
        })
        print("[liveTry] websocket connected; sent server_ready")

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                text = msg.get("text")
                if text is not None:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        print(f"[liveTry] bad json: {text[:120]!r}")
                        continue

                    msg_type = str(payload.get("type", "")).lower()
                    if msg_type == "start":
                        input_sr = int(payload.get("sample_rate", payload.get("sampleRate", input_sr)))
                        print(f"[liveTry] start: browser_sample_rate={input_sr}")
                    elif msg_type == "stop":
                        print("[liveTry] stop requested")
                        break
                    else:
                        print(f"[liveTry] text message: {payload}")
                    continue

                data = msg.get("bytes")
                if not data:
                    continue
                pcm_i16 = np.frombuffer(data, dtype=np.int16)
                if pcm_i16.size == 0:
                    continue

                packets += 1
                samples += int(pcm_i16.size)
                if packets == 1 or packets % 50 == 0:
                    pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(np.square(pcm_f32, dtype=np.float32))))
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    print(
                        "[liveTry] mic "
                        f"packets={packets} samples={samples} "
                        f"audio_sec={samples / max(float(input_sr), 1.0):.2f} "
                        f"wall_sec={elapsed:.2f} rms={rms:.5f}"
                    )

                moshi.append_browser_pcm(pcm_i16, input_sr)
                for ev in moshi.process_ready_steps():
                    await ws.send_json({
                        "type": "chunk_audio",
                        "chunk_id": ev["step"],
                        "sample_rate": ev["sample_rate"],
                        "pcm_s16le_b64": ev["reply_i16_b64"],
                        "gen_ms": ev["total_ms"],
                    })
                    # Two static frames per 80 ms Moshi step ~= 25 fps.
                    for frame_idx in range(2):
                        await ws.send_json({
                            "type": "chunk_frame",
                            "chunk_id": ev["step"],
                            "frame_idx": frame_idx,
                            "jpeg_b64": moshi.placeholder_jpeg_b64,
                            "server_fps": 25.0,
                            "chunks_done": ev["step"],
                            "avg_gen_ms": ev["total_ms"],
                            "moshi_text": ev["audio_text"] or ev["sampled_text"],
                        })
        except WebSocketDisconnect:
            pass
        finally:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            print(
                "[liveTry] websocket closed "
                f"packets={packets} samples={samples} "
                f"audio_sec={samples / max(float(input_sr), 1.0):.2f} wall_sec={elapsed:.2f}"
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--html_path", default=str(Path(__file__).resolve().parent / "static" / "index_v3.html"))
    parser.add_argument("--placeholder_path", default="")
    parser.add_argument("--moshi_root", default="/workspace/moshi")
    parser.add_argument("--mimi_hf_repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg_coef", type=float, default=1.0)
    args = parser.parse_args()

    app = build_app(args)

    import uvicorn

    print(f"[liveTry] serving {args.html_path}")
    print(f"[liveTry] open http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
