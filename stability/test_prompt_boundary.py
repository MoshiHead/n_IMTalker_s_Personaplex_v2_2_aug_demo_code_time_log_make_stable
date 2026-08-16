"""Regression tests built from conversation_logs_4.

That session had clean audio (total_dropped_speech_s=0.0), correct transcripts,
correct search results and correct compression -- and still answered
"What is Bitcoin?" with:

    "I'd say it's because they're a small team with a big vision. They've been
     working on AI stuff for years, they're constantly fine tuning the models"

The system prompt ("...promote RB Labs robots when relevant") is force-fed as
the assistant's OWN speech, and with --prompt_settle_sec 0 the model resumed
straight out of it, continuing the monologue it believed it had just delivered.
Because the KV cache is never reset per turn, that mode ran the whole session
and swallowed both retrieved facts.

Run from the repo root:  python stability/test_prompt_boundary.py
Needs no third-party packages.
"""
import ast
import pathlib
import re
import sys

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
LIVETRY = REPO / "IMTalker/liveTry.py"
SERVER = REPO / "IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
LAUNCHER = REPO / "IMTalker/start_winner_live.sh"
LM = REPO / "personaplex/moshi/moshi/models/lm.py"

livetry_src = LIVETRY.read_text(encoding="utf-8")
server_src = SERVER.read_text(encoding="utf-8")
launcher = LAUNCHER.read_text(encoding="utf-8")


def method_src(src, name):
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node)


def code_only(text):
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


# =========================== part A: the prompt must end at a turn boundary
print("-- part A: the model must leave the system prompt --")

settle = code_only(method_src(livetry_src, "_settle_after_prompt"))

# The whole point: pad in the CONVERSATIONAL register (silence on the user
# stream), not the PRIMING one (sine). Padding with sine keeps the model inside
# the prompt -- which is why long values used to mute it.
assert "PROMPT_SETTLE_USER_STREAM" in settle, "the register must be A/B-switchable"
assert '"silence"' in settle, "conversational silence must be the default register"
m = re.search(r"use_sine\s*=.*?==\s*\"sine\"", settle, re.S)
assert m, "sine must be opt-in, not the default"
print("[ok] 1. settle pads with conversational silence; sine is opt-in")

# Verify the premise against the fork itself: sine on the user stream appears
# only inside prompt priming, so it cannot signal 'the prompt is over'.
lm_src = LM.read_text(encoding="utf-8")
sine_sites = [ln for ln in lm_src.splitlines() if "_encode_sine_frame()" in ln]
assert len(sine_sites) == 3, f"expected 3 sine sites in the fork, found {len(sine_sites)}"
print("[ok] 2. premise holds: sine-on-user appears only in the fork's priming helpers")

# It must be ON by default now, and long enough to read as end of turn.
m = re.search(r'--prompt_settle_sec "\$\{PROMPT_SETTLE_SEC:-([\d.]+)\}"', launcher)
assert m, "the launcher must set a prompt_settle_sec default"
settle_default = float(m.group(1))
assert settle_default > 0.0, "the settle must be ON by default -- 0.0 is what logs_4 ran"
# This system calls 12 frames (0.96s) of silence an end-of-utterance; the settle
# should be at least that to read the same way to the model.
assert settle_default >= 0.9, f"{settle_default}s is shorter than this system's own 0.96s turn gap"
assert settle_default <= 3.0, f"{settle_default}s risks conditioning the model to stay silent"
print(f"[ok] 3. settle defaults to {settle_default}s, matching the 0.96s gap "
      f"this system already treats as end-of-utterance")

# A disabled settle must say so, since it is the failure mode of logs_4.
assert "prompt settle DISABLED" in method_src(livetry_src, "_settle_after_prompt")
print("[ok] 4. a disabled settle is announced in the log rather than passing silently")


# ================================= part B: no untrained tag syntax to speak
print("\n-- part B: tags only when the adapter that knows them is loaded --")

route = method_src(server_src, "_route_and_search")
for call, plain in (("wrap_with_ref_tags", "ref_content"),
                    ("wrap_with_lookup_tags", '"Please wait a minute."')):
    assert call in route, f"{call} should still be reachable"
    # Gated on the adapter having ACTUALLY loaded, not merely on a directory
    # having been passed: conversation_logs_5 recorded ref_lora_loaded=False
    # alongside injected <ref> tags, which is that distinction going wrong.
    seg = route[route.index(call):route.index(call) + 260]
    assert "self.ref_lora_active" in seg, f"{call} must be gated on ref_lora_active"
    assert plain in seg, f"{call} needs a plain-text alternative"
assert "self.ref_lora_dir else" not in route, \
    "the tag decision must not key off the requested dir"
print("[ok] 5. <ref> and <lookup> are used only when the adapter really loaded")
print("[ok] 6. otherwise plain text is injected -- no untrained syntax to read aloud")


# ======================================= part C: everything else still holds
print("\n-- part C: earlier fixes must not have regressed --")

checks = [
    (server_src, "IMTALKER_INJECT_USER_STREAM", "injection uses the conversational register"),
    (server_src, "WATCHDOG", "the thinking-sound watchdog is present"),
    (server_src, "self._stt_preroll", "the STT pre-roll window is present"),
    (livetry_src, "_input_dropped_speech_samples", "speech-vs-silence drop accounting"),
    (livetry_src, "seed_personaplex", "PersonaPlex seeding"),
    (launcher, 'REF_LORA_ENABLED="${REF_LORA_ENABLED:-1}"', "ref LoRA enabled by default"),
    (launcher, "WEB_SEARCH_MIN_SCORE:-0.50", "relevance floor still 0.50"),
]
for src, needle, label in checks:
    assert needle in src, f"regression: {label}"
    print(f"[ok] {label}")

# The background thread still must not own search state.
assigns = re.findall(r"self\.(search_awaiting_ref|search_ref_committed_this_turn)\s*=", route)
assert not assigns, f"background thread must not assign {set(assigns)}"
print("[ok] background thread still owns only the pending_* handoff slots")

print("\nAll prompt-boundary checks passed.")
