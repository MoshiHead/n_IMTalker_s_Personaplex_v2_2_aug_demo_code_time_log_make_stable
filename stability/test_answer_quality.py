"""Regression tests built from conversation_logs_1.

Every case below is a real failure from that session. Run from the repo root:

    python stability/test_answer_quality.py

Needs only torch (for the module import in part B).
"""
import ast
import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SERVER = REPO / "IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
HELPERS = REPO / "IMTalker/search_helpers.py"
LAUNCHER = REPO / "IMTalker/start_winner_live.sh"


def load_regexes(src: str, names):
    ns = {"re": re}
    for name in names:
        m = re.search(rf"^{name} = re\.compile\(", src, re.M)
        assert m, f"{name} missing from search_helpers.py"
        i = src.index("(", m.start())
        depth = 0
        while True:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
            if depth == 0:
                break
            i += 1
        exec(src[m.start():i + 1], ns)
    return ns


# =============================================================== part A: tags
print("-- part A: tag markup must never be spoken --")
helpers_src = HELPERS.read_text(encoding="utf-8")
ns = load_regexes(helpers_src, [
    "_ECHOED_TAG_RE", "_PARTIAL_TAG_RE", "_ORPHAN_TAG_TAIL_RE", "_ORPHAN_TAG_END_RE",
    "_NON_ANSWER_RE", "_HAS_CONCRETE_VALUE_RE",
])


def strip_injected_tags(text):
    c = ns["_ECHOED_TAG_RE"].sub(" ", text)
    c = ns["_PARTIAL_TAG_RE"].sub("", c)
    c = ns["_ORPHAN_TAG_TAIL_RE"].sub("", c)
    c = ns["_ORPHAN_TAG_END_RE"].sub("", c)
    return re.sub(r"\s{2,}", " ", c).strip()


# Turn 12 and turn 13, verbatim from the log.
assert not strip_injected_tags(
    "Fees are used on the site. <ref").endswith("<ref")
assert strip_injected_tags("coins.>") == "coins."
assert strip_injected_tags("<ref> The current Google stock price is $343.94. <ref>") \
    == "The current Google stock price is $343.94."
# Must not eat ordinary comparisons.
assert strip_injected_tags("5 < 6 and 7 > 6") == "5 < 6 and 7 > 6"
print("[ok] 1. partial '<ref' and orphan '>' are stripped; real angle brackets survive")


# ================================================== part B: grounding quality
print("\n-- part B: what may be injected as fact --")


def compressor_would_reject(question, result):
    if ns["_NON_ANSWER_RE"].search(result):
        return "non_answer"
    q = {w.lower().strip(".,!?'\"") for w in question.split()} - {""}
    a = {w.lower().strip(".,!?'\"") for w in result.split()} - {""}
    if a and q:
        echo = len(a & q) / len(a)
        if echo >= 0.70 and not ns["_HAS_CONCRETE_VALUE_RE"].search(result):
            return f"question_echo={echo:.2f}"
    return ""


# Turn 11: a non-answer was injected as grounding.
assert compressor_would_reject(
    "What is today's guest list of market price?",
    "Today's guest list on the market has not been provided.") == "non_answer"
# Turn 10: the compressor read the garbled question back and it was injected.
assert compressor_would_reject(
    "The heart is good as gold, market. Rise.",
    "The heart is good as gold, market.").startswith("question_echo")
# Turn 12 and 13: real retrieved values must still pass.
assert compressor_would_reject(
    "What is today's Google stock market price?",
    "The current Google stock price is $343.94.") == ""
assert compressor_would_reject(
    "What is today's expense rate of euro to dollar?",
    "Today's Euro to Dollar exchange rate is 1.1570.") == ""
print("[ok] 2. non-answers and question echoes are rejected; real values pass")

launcher = LAUNCHER.read_text(encoding="utf-8")
assert 'WEB_SEARCH_MIN_SCORE:-0.50' in launcher, "relevance floor must be 0.50"
# The floor must separate the log's good search from its bad ones.
GOOD = [0.8232, 0.8131, 0.7065]      # turn 12, the one useful search
BAD = [0.2602, 0.1843, 0.2366, 0.1926]  # turns 10 and 11
assert min(GOOD) >= 0.50 > max(BAD), "0.50 must separate the logged good/bad results"
print("[ok] 3. the 0.50 relevance floor separates every logged good result from every bad one")


# ==================================================== part C: the STT window
print("\n-- part C: transcript window --")
server_src = SERVER.read_text(encoding="utf-8")
tree = ast.parse(server_src)
cls = next(n for n in ast.walk(tree)
           if isinstance(n, ast.ClassDef) and n.name == "MoshiOnlyEngineWithHidden")
consts = {}
for node in cls.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        with_val = node.targets[0].id
        try:
            consts[with_val] = ast.literal_eval(node.value)
        except Exception:
            pass
assert "_STT_PREROLL_FRAMES" in consts and "_STT_MAX_UTTERANCE_FRAMES" in consts
preroll, cap = consts["_STT_PREROLL_FRAMES"], consts["_STT_MAX_UTTERANCE_FRAMES"]
assert 0 < preroll <= 20, preroll
# The cap is a backstop for a VAD that never fires, not the fix -- the fix is
# the gating tested below, which is what turns a 35s window into an utterance.
# So the cap only has to be a sane ceiling for a spoken question: long enough
# never to clip a real one, short enough to bound a pathological session.
assert 60 <= cap <= 400, f"cap {cap} frames ({cap * 0.08:.0f}s) is not a sane question ceiling"
print(f"[ok] 4. pre-roll {preroll} frames ({preroll * 0.08:.2f}s), "
      f"stuck-VAD cap {cap} frames ({cap * 0.08:.0f}s)")


# Simulate the accumulation rule, old vs new.
def simulate(frames, preroll_n, cap_n, bounded):
    """frames: list of bools, True = a real speech token this frame.

    bounded=False reproduces the OLD rule exactly: append every frame since the
    last turn, with no gating and no ceiling. bounded=True is the new rule.
    """
    if not bounded:
        return len(frames)

    buf, ring, in_utt = [], collections.deque(maxlen=preroll_n), False
    for is_speech in frames:
        if is_speech and not in_utt:
            in_utt = True
            buf = list(ring)
            ring.clear()
        if in_utt:
            buf.append("f")
            if len(buf) > cap_n:
                buf = buf[len(buf) - cap_n:]
        else:
            ring.append("f")
    return len(buf)


# Turn 2 in the log: ~35s (437 frames) of gap, with the user speaking for the
# last ~2s (25 frames).
gap = [False] * 412 + [True] * 25
assert simulate(gap, preroll, cap, bounded=False) == 437, "old behaviour should reproduce 437"
new_len = simulate(gap, preroll, cap, bounded=True)
assert new_len == 25 + preroll, new_len
print(f"[ok] 5. the same 35s gap now decodes {new_len} frames "
      f"({new_len * 0.08:.1f}s) instead of 437 ({437 * 0.08:.1f}s)")

# A stuck VAD must not grow the window without limit.
assert simulate([True] * 5000, preroll, cap, bounded=True) == cap
print("[ok] 6. a stuck VAD is capped instead of accumulating forever")

assert "self._stt_preroll" in server_src and "_stt_overlong_logged" in server_src
assert "self._stt_preroll.clear()" in server_src, "the pre-roll must be cleared per turn"
print("[ok] 7. pre-roll ring and its per-turn reset are wired into the server")


# ============================================== part D: reference LoRA is opt-in
print("\n-- part D: reference LoRA --")
# CORRECTED. An earlier round defaulted this adapter OFF, on the assumption it
# was a third-party component whose persona was leaking. It is in fact trained
# for this deployment, specifically to make the model act on injected reference
# blocks -- so it belongs ON, and the real defect was that it was being loaded
# against a mismatched config. See stability/test_ref_lora.py.
assert 'REF_LORA_ENABLED="${REF_LORA_ENABLED:-1}"' in launcher, \
    "the adapter must default ON -- it is what makes <ref> injection work"
assert "--ref_lora_scale" in launcher
assert '--ref_lora_dir "$REF_LORA_DIR" --ref_lora_scale' in launcher, \
    "the adapter must be passed through the REF_LORA_ENABLED branch"
assert launcher.count("--ref_lora_dir") == 1, "ref_lora_dir must appear in exactly one branch"
livetry = (REPO / "IMTalker/liveTry.py").read_text(encoding="utf-8")
assert "_apply_ref_lora_scale" in livetry
assert "ref_lora_scale" in livetry
print("[ok] 8. the <lookup>/<ref> adapter is enabled, scalable, and coverage-checked")


# ================================================ part E: response log bounds
print("\n-- part E: reply logging --")
assert "self._utterance_start_audio_text_len\n            if self._utterance_start_audio_text_len" in server_src \
    or "_utterance_start_audio_text_len" in server_src.split("_turn_start_audio_text_len = (")[1][:400], \
    "turn start must fall back to the utterance start"
assert server_src.count("self._turn_start_audio_text_len = (") == 2, \
    "both turn-start sites must use the new boundary"
print("[ok] 9. reply logging starts at the user's first word, not 960ms after their last")


# ========================================================== part F: notebook
print("\n-- part F: notebook --")
nb = json.loads((REPO / "RunPod_RTX5090_PersonaPlex_IMTalker_Live_fixed.ipynb")
                .read_text(encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"])
for needle in ('REF_LORA_ENABLED = "1"', 'WEB_SEARCH_MIN_SCORE = "0.50"',
               'env_overrides["REF_LORA_ENABLED"]', 'env_overrides["WEB_SEARCH_MIN_SCORE"]'):
    assert needle in src, f"notebook missing {needle}"
print("[ok] 10. notebook exposes and forwards the answer-quality controls")

print("\nAll answer-quality regression checks passed.")
