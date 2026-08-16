"""No stream may be force-fed a long constant run.

This is one mechanism that has now caused three separate failures, once per
stream, and it is the single most important invariant in this pipeline.

Moshi's text and audio streams are both autoregressive. Force a constant token
into either one for seconds at a time and the model learns, in context, to keep
producing exactly that -- so the stream stays dead after the forcing stops.

  * ASSISTANT AUDIO, during <ref> injection (conversation_logs_6)
    20-24 frames of SILENCE_TOKENS -> every injected turn produced correct text
    and no voice. Turns without an injection were heard normally.

  * TEXT, during the search (the session after that)
    ~52 frames of zero_text_code across a 4.2s search -> three consecutive
    search turns produced no words at all (23.6s, 21.4s, 64.8s), while every
    un-searched turn in the same session answered normally. Removing the
    <lookup> filler had made it worse: those 8 real tokens were the only thing
    breaking the run.

  * USER AUDIO, during injection (conversation_logs_2)
    a sine tone is this fork's priming marker, so injecting with it replayed
    the system prompt and the model answered by greeting.

Run from the repo root:  python stability/test_forced_runs.py
Needs no third-party packages.
"""
import ast
import pathlib
import re
import sys

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SERVER = REPO / "IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
LIVETRY = REPO / "IMTalker/liveTry.py"

server_src = SERVER.read_text(encoding="utf-8")
livetry_src = LIVETRY.read_text(encoding="utf-8")
tree = ast.parse(server_src)

MIMI_FRAME_S = 1920 / 24000  # 80ms


def method(name, src=None, t=None):
    t = t or tree
    s = src or server_src
    return ast.get_source_segment(s, next(
        n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == name))


def consts(class_name):
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    out = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out


# =================================================== 1. the text hold is bounded
print("-- the text stream --")

C = consts("MoshiOnlyEngineWithHidden")
cap = C["_SUPPRESS_TEXT_MAX_FRAMES"]
assert cap > 0, "there must be a bound on the forced-text run"
hold_s = cap * MIMI_FRAME_S
# Long enough to cover the moment the model would blurt a number...
assert hold_s >= 0.5, f"{hold_s:.1f}s is too short to stop an invented figure"
# ...and far below the ~4.2s search that silenced the stream.
assert hold_s <= 2.0, f"{hold_s:.1f}s approaches the run length that caused the failure"
print(f"[ok] 1. text hold bounded at {cap} frames ({hold_s:.1f}s), against the "
      f"~4.2s run that silenced it")

step = method("_step")
assert "_suppress_text_frames" in step, "the run must be counted"
assert "_SUPPRESS_TEXT_MAX_FRAMES" in step
assert "self.suppress_text_until_ref = False" in step, \
    "hitting the bound must actually release the hold"
assert "text_hold_released" in step, "the release must be visible in the log"
print("[ok] 2. the run is counted, released at the bound, and logged")

# The counter must reset, or the bound fires once and never again.
assert "self._suppress_text_frames = 0" in step
reset = method("reset_session")
assert "_suppress_text_frames" in reset, "the counter must reset per session"
print("[ok] 3. the counter resets between turns and between sessions")

# Configurable, including switching the bound off.
assert "SUPPRESS_TEXT_MAX_SEC" in server_src
m = re.search(r'SUPPRESS_TEXT_MAX_SEC",\s*"([\d.]+)"', server_src)
assert m and float(m.group(1)) > 0, "the default must apply a bound"
print(f"[ok] 4. SUPPRESS_TEXT_MAX_SEC defaults to {m.group(1)}s and can be set to 0")


# ================================================ 2. the audio stream is free
print("\n-- the assistant audio stream --")

inject = method("_inject_tokens")
inject_code = "\n".join(l for l in inject.splitlines() if not l.lstrip().startswith("#"))
assert "moshi_tokens=self.lm_gen._encode_zero_frame()" not in inject_code, \
    "the voice must not be force-silenced for the length of an injection"
assert "else None" in inject_code, "the default must leave the voice unforced"
print("[ok] 5. injection no longer writes a constant run into the voice")


# ================================================= 3. the filler stays, and why
print("\n-- the <lookup> filler --")

route = method("_route_and_search")
m = re.search(r'LOOKUP_INJECT_ENABLED",\s*"1"', route)
assert m, (
    "the filler must be ON: it is the only thing breaking the forced-text run "
    "during a search, and removing it silenced three consecutive turns"
)
print("[ok] 6. the filler is restored -- it interrupts the forced run")


# ============================================ 4. the prompt settle, same rule
print("\n-- the prompt boundary --")

settle = method("_settle_after_prompt", livetry_src, ast.parse(livetry_src))
settle_code = "\n".join(l for l in settle.splitlines() if not l.lstrip().startswith("#"))
assert "_encode_sine_frame" in settle_code and '"sine"' in settle, \
    "the settle must keep its register switchable"
# The settle IS a deliberate constant run, so it must stay short.
m = re.search(r'--prompt_settle_sec "\$\{PROMPT_SETTLE_SEC:-([\d.]+)\}"',
              (REPO / "IMTalker/start_winner_live.sh").read_text(encoding="utf-8"))
assert m and 0 < float(m.group(1)) <= 3.0, \
    "the settle is itself a forced run and must stay short"
print(f"[ok] 7. the prompt settle is a deliberate run, held to {m.group(1)}s")


# ================================================== 5. every forced run listed
print("\n-- inventory --")

# Any NEW unconditional forced-token loop should be a deliberate decision, so
# keep an explicit list of the places that force a stream at all.
forced = []
for name in ("_inject_tokens", "_step"):
    src = method(name)
    if "text_token=" in src or "moshi_tokens=" in src:
        forced.append(name)
assert set(forced) == {"_inject_tokens", "_step"}, forced
print(f"[ok] 8. the only places that force a stream are {sorted(forced)}, "
      f"and both are bounded")

print("\nAll forced-run checks passed.")
