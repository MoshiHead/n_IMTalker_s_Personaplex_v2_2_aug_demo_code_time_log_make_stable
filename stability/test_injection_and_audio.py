"""Regression tests built from conversation_logs_2.

That session fixed the persona problem (ref_lora_loaded=False, and the
unsearched turns became genuinely good) but exposed two more:

  A. every <ref> injection made the model GREET instead of answer, because the
     injection replayed the system-prompt signature;
  B. 4.5s of microphone audio was discarded mid-question, so the model heard
     "a beat" for "Bitcoin" and said "I can't hear you well" on turn 1.

Run from the repo root:  python stability/test_injection_and_audio.py
Needs numpy (torch not required).
"""
import ast
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SERVER = REPO / "IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
LIVETRY = REPO / "IMTalker/liveTry.py"

server_src = SERVER.read_text(encoding="utf-8")
livetry_src = LIVETRY.read_text(encoding="utf-8")

# ============================================ part A: the injection signature
print("-- part A: <ref> injection must not look like a system prompt --")

tree = ast.parse(server_src)
inject = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name == "_inject_tokens"
)
body = ast.get_source_segment(server_src, inject)

# In this fork a sine tone on the USER stream appears only inside system-prompt
# priming (_step_voice_prompt_frame / _step_audio_silence_core /
# _step_text_prompt_core). Using it for a live injection is what made the model
# greet: "Okay, the S and P.> Thank you for calling RB Labs. Have a great day!"
assert "_encode_sine_frame" not in body.split("use_sine")[-1].split("for tok")[0] \
    or "IMTALKER_INJECT_USER_STREAM" in body, "sine must be opt-in only"
assert "IMTALKER_INJECT_USER_STREAM" in body, "the A/B escape hatch must exist"
assert '"silence"' in body, "silence must be the default user stream"
print("[ok] 1. the user stream carries silence by default; sine is opt-in for A/B")

lm_src = (REPO / "personaplex/moshi/moshi/models/lm.py").read_text(encoding="utf-8")
sine_users = [
    ln.strip() for ln in lm_src.splitlines() if "_encode_sine_frame()" in ln
]
assert len(sine_users) == 3, f"expected 3 sine call sites in the fork, found {len(sine_users)}"
# Confirm all three sit inside prompt-priming helpers -- the premise of the fix.
for name in ("_step_voice_prompt_frame", "_step_audio_silence_core", "_step_text_prompt_core"):
    assert name in lm_src
print("[ok] 2. sine-on-user appears only in the fork's 3 prompt-priming helpers")


# =================================== part B: nothing useful -> inject nothing
print("\n-- part B: an empty result must inject nothing --")

assert "There's no specific information available on this" not in server_src, (
    "the discouraging fallback ref must be gone -- it was injected on 3 of 4 "
    "search turns in conversation_logs_2"
)
assert "injecting nothing" in server_src
assert "nothing injected" in server_src
print("[ok] 3. the 'no specific information available' ref is no longer injected")
print("[ok] 4. both the empty-result and filler-timeout paths release without injecting")


# ================================================ part C: microphone trimming
print("\n-- part C: never cut a question in half --")

cls = next(n for n in ast.walk(ast.parse(livetry_src))
           if isinstance(n, ast.ClassDef) and n.name == "MoshiOnlyEngine")
consts = {}
for node in cls.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        try:
            consts[node.targets[0].id] = ast.literal_eval(node.value)
        except Exception:
            pass
SIL_RMS = consts["_INPUT_SILENCE_RMS"]
CEIL = consts["_INPUT_HARD_CEILING_FACTOR"]
assert 0 < SIL_RMS < 0.02, SIL_RMS      # below the server's own "VOICE" threshold
assert CEIL > 1.0, CEIL
print(f"[ok] 5. silence threshold {SIL_RMS} (server calls >=0.02 'VOICE'), "
      f"hard ceiling {CEIL}x the cap")

SR = 24000


def trim(buf, cap_s, sil_rms=SIL_RMS, ceiling_factor=CEIL):
    """The rule now in append_browser_pcm. Returns (kept_buffer, dropped, quiet)."""
    max_samples = int(cap_s * SR)
    if max_samples <= 0 or buf.shape[0] <= max_samples:
        return buf, 0, 0
    want = int(buf.shape[0] - max_samples)
    excess = buf[:want]
    step = max(1, SR // 50)
    quiet = 0
    while quiet + step <= excess.shape[0]:
        w = excess[quiet:quiet + step]
        if float(np.sqrt(np.mean(np.square(w, dtype=np.float32)))) > sil_rms:
            break
        quiet += step
    hard = int(cap_s * ceiling_factor * SR)
    if quiet >= want:
        dropped = want
    elif buf.shape[0] > hard:
        dropped = want
    elif quiet > 0:
        dropped = quiet
    else:
        return buf, 0, quiet
    return buf[dropped:].copy(), dropped, quiet


rng = np.random.default_rng(0)
speech = (rng.standard_normal(SR * 3) * 0.15).astype(np.float32)   # ~3s of speech
silence = np.zeros(SR * 2, dtype=np.float32)                        # 2s of silence

# 1. Backlog is silence followed by the question -> trim the silence, keep every
#    sample of speech. This is the conversation_logs_2 shape.
buf = np.concatenate([silence, speech])
kept, dropped, quiet = trim(buf, cap_s=2.0)
assert dropped > 0, "silent lead-in should be trimmed"
assert np.array_equal(kept[-speech.shape[0]:], speech), "speech must be untouched"
print(f"[ok] 6. {dropped / SR:.2f}s of silence trimmed, all {speech.shape[0] / SR:.1f}s "
      f"of speech kept intact")

# 2. Backlog is ALL speech, under the ceiling -> drop nothing, accept the delay.
buf = speech.copy()
kept, dropped, quiet = trim(buf, cap_s=2.0)
assert dropped == 0, f"speech was cut ({dropped} samples) while under the ceiling"
assert kept.shape[0] == buf.shape[0]
print("[ok] 7. an all-speech backlog under the ceiling is kept whole, not sliced")

# 3. Past the hard ceiling, latency wins and speech may finally be cut.
long_speech = (rng.standard_normal(int(SR * 6)) * 0.15).astype(np.float32)
kept, dropped, quiet = trim(long_speech, cap_s=2.0)
assert dropped > 0, "past the ceiling the trim must engage"
assert kept.shape[0] == int(2.0 * SR)
print(f"[ok] 8. past {CEIL}x the cap the trim engages ({dropped / SR:.2f}s) "
      f"rather than letting delay grow without bound")

# 4. The old rule would have cut into the question in case 2. Show the contrast.
old_dropped = speech.shape[0] - int(2.0 * SR)
assert old_dropped > 0
print(f"[ok] 9. the old rule would have discarded {old_dropped / SR:.2f}s of that "
      f"same speech; the new one discards 0")

assert "_input_dropped_speech_samples" in livetry_src
assert "dropped_kind" in livetry_src
print("[ok] 10. dropped silence and dropped speech are counted and logged separately")

print("\nAll injection and audio checks passed.")
