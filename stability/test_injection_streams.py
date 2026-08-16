"""Which stream each injection argument writes, and why that made the avatar mute.

conversation_logs_6 is the clean experiment. Turns WITHOUT a <ref> injection
were heard normally; all three turns WITH one produced correct text and no
voice at all:

    turn 1  "So Bitcoin, it's digital money..."              no injection -> HEARD
    turn 2  "The current gold price is $140.72 per gram."    injection    -> mute
    turn 3  (no spoken response in 20.9s)                    injection    -> mute
    turn 4  "Good day."                                      injection    -> mute
    turn 5  "You're welcome. Have a great day."              no injection -> HEARD

`_inject_tokens` forced SILENCE_TOKENS into `moshi_tokens` for every injected
token. This test pins, from the fork's own source, that `moshi_tokens` is the
ASSISTANT's audio stream -- so that was 1.6-1.9 seconds of digital silence
written into the voice while real words went into the text stream. The audio
stream is autoregressive, so it carried on silent.

Run from the repo root:  python stability/test_injection_streams.py
Needs no third-party packages.
"""
import ast
import pathlib
import re
import sys

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
LM = REPO / "personaplex/moshi/moshi/models/lm.py"
SERVER = REPO / "IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"

lm_src = LM.read_text(encoding="utf-8")
server_src = SERVER.read_text(encoding="utf-8")


# ================== part A: establish the stream mapping from the fork itself
print("-- part A: which codebooks does each argument write? --")

prep = lm_src[lm_src.index("def prepare_step_input"):]
prep = prep[:prep.index("def step(")]

# moshi_tokens -> codebooks 1..8
assert re.search(r"for q_moshi in range\(moshi_tokens\.shape\[1\]\):\s*\n\s*k = 1 \+ q_moshi", prep), \
    "moshi_tokens must map to codebooks starting at 1"
# input_tokens -> codebooks 9..16
assert re.search(
    r"for q_other in range\(input_tokens\.shape\[1\]\):\s*\n\s*k = AUDIO_TOKENS_PER_STREAM \+ 1 \+ q_other",
    prep), "input_tokens must map to codebooks starting at AUDIO_TOKENS_PER_STREAM+1"
print("[ok] 1. moshi_tokens -> codebooks 1..8, input_tokens -> codebooks 9..16")

# The decode path proves codebooks 1.. are the assistant's voice.
assert "self.mimi.decode(tokens[:, 1:])" in server_src, \
    "the reply audio is decoded from codebook 1 onwards"
# And the live step passes the MIC codes as input_tokens (first positional).
assert "self.lm_gen._step(codes[:, :, :1])" in server_src
print("[ok] 2. the server decodes codebooks 1.. as the reply -> moshi_tokens IS the voice")
print("[ok] 3. the server feeds mic codes as input_tokens -> that is the user stream")


# ============================ part B: the injection must not silence the voice
print("\n-- part B: the injection --")

tree = ast.parse(server_src)
inject = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_inject_tokens")
body = ast.get_source_segment(server_src, inject)
code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))

# The old line, verbatim, must be gone.
assert "moshi_tokens=self.lm_gen._encode_zero_frame()" not in code, (
    "the assistant's audio stream must not be forced to SILENCE_TOKENS -- that is "
    "what made every injected turn mute"
)
assert "moshi_tokens=assistant_frame" in code
assert "IMTALKER_INJECT_ASSISTANT_STREAM" in code, "the old behaviour must stay A/B-switchable"
print("[ok] 4. the assistant stream is no longer force-silenced during injection")

# Default must be 'generate' (None passed), with 'silence' as the opt-in.
m = re.search(r'IMTALKER_INJECT_ASSISTANT_STREAM",\s*"generate"', code)
assert m, "generating the assistant's own audio must be the default"
assert re.search(r"assistant_frame\s*=\s*self\.lm_gen\._encode_zero_frame\(\)\s*if\s*force_assistant_silence\s*else\s*None",
                 code), "the non-forced path must pass None, i.e. 'not provided'"
print("[ok] 5. default passes None -> the depformer samples the voice as during normal speech")

# The user stream keeps its own (separately established) fix.
assert "input_tokens=user_frame" in code
assert "IMTALKER_INJECT_USER_STREAM" in code
print("[ok] 6. the user stream still carries silence, not the priming sine tone")


# ===================================== part C: None really means 'not provided'
print("\n-- part C: None is a valid 'do not force' --")

assert "if moshi_tokens is not None:" in prep, \
    "prepare_step_input must treat None as 'not provided' rather than asserting"
# ...and unprovided audio codebooks are then sampled.
proc = lm_src[lm_src.index("def process_transformer_output"):]
proc = proc[:proc.index("def load_voice_prompt")]
assert "sampled_audio_tokens" in proc and "~state.provided" in proc, \
    "unprovided audio positions must be filled from sampled tokens"
print("[ok] 7. None leaves the positions unprovided, so they are filled by sampling")

print("\nAll injection-stream checks passed.")
