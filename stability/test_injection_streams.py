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


# ======================= part D: the filler must not be read aloud (logs_7)
print("\n-- part D: the <lookup> filler --")

route = next(n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "_route_and_search")
route_src = ast.get_source_segment(server_src, route)

# conversation_logs_7 turn 6 spoke ONLY the filler:
#     SAID "Please wait a minute."
# and turn 7 spoke it before the answer:
#     SAID "Please wait a minute The current Tesla stock is $309.32."
# Once the assistant's audio stream was freed (r10) the model voices whatever
# it is fed, so feeding it a sentence makes it say that sentence.
# CORRECTED. Turning this off, on the strength of the logs_7 echo, made the
# next session strictly worse: three consecutive search turns produced no words
# at all. Those 8 real tokens were the only thing interrupting a ~4.2s run of
# forced zero_text_code, and without them the text stream was conditioned into
# silence exactly as the audio stream had been. The echo was a cosmetic
# complaint; the silence was a broken feature. See stability/test_forced_runs.py.
assert "LOOKUP_INJECT_ENABLED" in route_src, "the filler injection must be switchable"
m = re.search(r'LOOKUP_INJECT_ENABLED",\s*"1"', route_src)
assert m, "the filler must default ON -- it breaks the forced-text run"
assert "self.pending_lookup_tokens = None" in route_src, \
    "the disabled branch must queue nothing"
print("[ok] 8. the <lookup> filler is on, breaking the forced-text run during a search")

# The wait is covered by the thinking sound; the text hold is now bounded.
assert "_start_thinking_sound" in server_src and "suppress_text_until_ref" in server_src
assert "_SUPPRESS_TEXT_MAX_FRAMES" in server_src, "the text hold must be bounded"
print("[ok] 9. the wait is covered by the thinking sound, with a bounded text hold")


# =================== part E: tell 'never spoke' from 'never delivered' (logs_7)
print("\n-- part E: which side of the pipeline failed --")

step = ast.get_source_segment(server_src, next(
    n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_step"))
assert "AUDIO NOT DELIVERED" in step and "MUTE AFTER INJECTION" in step, \
    "the watchdog must name both failure modes separately"
assert "audio_packets_enqueued" in server_src
assert "self._post_inject_audio_pkts0" in step, \
    "the packet count must be snapshotted at injection time"
print("[ok] 10. the watchdog separates 'model never spoke' from 'audio never delivered'")

# The counter has to be incremented where packets really leave the engine.
assert "reply_engine.audio_packets_enqueued += 1" in server_src
enq = server_src[server_src.index("def _enqueue_audio"):]
enq = enq[:enq.index("if prebuffer_chunks")]
assert "audio_packets_enqueued += 1" in enq, "count inside _enqueue_audio, not elsewhere"
print("[ok] 11. the counter increments inside _enqueue_audio itself")

print("\nAll injection-stream checks passed.")
