"""Regression tests built from conversation_logs_3.

That session confirmed the injection fix worked -- grounded answers came out
correct ("Today's gold price is $140.72 per gram") -- and exposed one more
fault, introduced by the previous round:

  tavily returned 0 results, the background thread cleared `search_awaiting_ref`
  itself, and the GPU thread therefore never ran the cancel path. No
  thinking_sound_stop was ever written; the clip looped for 33 seconds, echoed
  into the microphone, was transcribed as "Hello, it's Dolph. It's Dolph...",
  and poisoned the next session's opening context ("What is Bitcoin?" answered
  with "big planes are huge for transporting lots of people").

Run from the repo root:  python stability/test_search_state.py
Needs no third-party packages.
"""
import ast
import pathlib
import re
import sys

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SERVER = REPO / "IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
HELPERS = REPO / "IMTalker/search_helpers.py"

server_src = SERVER.read_text(encoding="utf-8")
tree = ast.parse(server_src)


def method_src(name):
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(server_src, node)


# ================================ part A: thread ownership of the search flags
print("-- part A: only the GPU thread may retire a search --")

route = method_src("_route_and_search")

# `search_awaiting_ref` is what makes the GPU thread call _consume_pending at
# all. The background thread must never clear it, or the cancel path -- which
# is what stops the thinking sound and releases the text hold -- never runs.
assigns = re.findall(r"self\.(search_awaiting_ref|search_ref_committed_this_turn)\s*=", route)
assert not assigns, (
    f"_route_and_search runs on the BACKGROUND thread and must not assign {set(assigns)}; "
    f"it may only write the handoff slots (pending_*)"
)
print("[ok] 1. _route_and_search never clears search_awaiting_ref / _ref_committed")

# It must still signal, via the handoff slot the GPU thread polls.
assert "self.pending_search_cancelled = True" in route
print("[ok] 2. it signals the GPU thread through pending_search_cancelled instead")

consume = method_src("_consume_pending")
assert "self.pending_search_cancelled" in consume
assert "_stop_thinking_sound" in consume
assert "self.suppress_text_until_ref = False" in consume
print("[ok] 3. the GPU thread's cancel path stops the sound and releases the text hold")


# ============================================ part B: the watchdog backstop
print("\n-- part B: a filler clip must never outlive its search --")

step = method_src("_step")
assert "WATCHDOG" in step, "the watchdog must live in _step, which runs every chunk"
assert "search_thinking_active" in step

# Check the CODE, not the comments -- the comment block deliberately names the
# stuck flag while explaining the bug.
step_code = "\n".join(
    ln for ln in step.splitlines() if not ln.lstrip().startswith("#")
)
# It has to be independent of the flags it protects, or it cannot catch the
# failure: being stuck IS the failure. Two properties prove that:
#   - its condition is search_thinking_active ALONE (no `and`), and
#   - it sits at the top level of _step, so it runs on every chunk rather than
#     nested inside a branch that a stuck flag could skip.
# Exactly this line -- the other `search_thinking_active` test in _step is the
# audio swap, which is correctly compounded with `thinking_sound_pcm is not None`.
guard_lines = [ln for ln in step_code.splitlines()
               if ln.strip() == "if self.search_thinking_active:"]
assert len(guard_lines) == 1, f"expected one bare watchdog guard, found {guard_lines}"
guard_line = guard_lines[0]
body_indent = len(step_code.splitlines()[1]) - len(step_code.splitlines()[1].lstrip())
assert len(guard_line) - len(guard_line.lstrip()) == body_indent, \
    "the watchdog must sit at the top level of _step, not nested in another branch"
watchdog_body = step_code.split(guard_line, 1)[1]
assert "playing_s" in watchdog_body and "budget_s" in watchdog_body
for released in ("self.suppress_text_until_ref = False",
                 "self.search_awaiting_ref = False",
                 "self.pending_ref_tokens = None"):
    assert released in step, f"watchdog must clear {released}"
print("[ok] 4. the watchdog keys only off the clip being active, and clears every flag")

# 33s of looping is what the log recorded; the budget must be far below that.
m = re.search(r"budget_s\s*=\s*\(self\._SEARCH_MAX_FILLER_FRAMES \* MIMI_FRAME_SIZE / TARGET_SR\)\s*\+\s*([\d.]+)", step)
assert m, "watchdog budget must derive from the configured filler cap"
margin = float(m.group(1))
# Defaults: --search_max_filler_sec 6.0 -> 6.0s + margin, comfortably under 33s.
budget = 6.0 + margin
assert budget < 15.0, f"watchdog budget {budget}s is too generous"
print(f"[ok] 5. budget is the filler cap + {margin}s = {budget}s at defaults, "
      f"versus the 33s hang that was logged")


# ================================================== part C: the '>' fragment
print("\n-- part C: no tag debris in speech --")
helpers_src = HELPERS.read_text(encoding="utf-8")
ns = {"re": re}
for name in ("_ECHOED_TAG_RE", "_PARTIAL_TAG_RE", "_ORPHAN_TAG_TAIL_RE",
             "_ORPHAN_TAG_END_RE", "_ORPHAN_TAG_MID_RE"):
    m = re.search(rf"^{name} = re\.compile\(", helpers_src, re.M)
    assert m, name
    i = helpers_src.index("(", m.start())
    depth = 0
    while True:
        if helpers_src[i] == "(":
            depth += 1
        elif helpers_src[i] == ")":
            depth -= 1
        if depth == 0:
            break
        i += 1
    exec(helpers_src[m.start():i + 1], ns)


def strip(text):
    c = ns["_ECHOED_TAG_RE"].sub(" ", text)
    c = ns["_PARTIAL_TAG_RE"].sub("", c)
    c = ns["_ORPHAN_TAG_TAIL_RE"].sub("", c)
    c = ns["_ORPHAN_TAG_END_RE"].sub("", c)
    c = ns["_ORPHAN_TAG_MID_RE"].sub(" ", c)
    return re.sub(r"\s{2,}", " ", c).strip()


# Verbatim from conversation_logs_3.
assert strip("Hmm, the stock market.> Today's gold price is $140.72 per gram.") \
    == "Hmm, the stock market. Today's gold price is $140.72 per gram."
assert strip("Hmm, the Tesla.> As of today, Tesla's stock is valued at $341.64 per share.") \
    == "Hmm, the Tesla. As of today, Tesla's stock is valued at $341.64 per share."
assert strip("Hmm, the exchange rate.> The current exchange rate of 1 Euro to 1.15 US Dollars.") \
    == "Hmm, the exchange rate. The current exchange rate of 1 Euro to 1.15 US Dollars."
# Ordinary comparisons must survive: no full stop before the bracket.
assert strip("5 > 3 is true") == "5 > 3 is true"
assert strip("if x >= 3 then") == "if x >= 3 then"
print("[ok] 6. mid-reply '.>' debris is removed; real comparisons survive")


# ====================================== part D: what counts as the answer
print("\n-- part D: a grounded reply is what came AFTER the facts --")
assert "self._turn_start_audio_text_len = len(self.audio_text)" in consume, \
    "the reply boundary must be reset at injection time"
idx_inject = consume.index("self._inject_tokens(ref_tokens)")
idx_bound = consume.index("self._turn_start_audio_text_len = len(self.audio_text)")
assert idx_bound > idx_inject, "the boundary must move AFTER the injection, not before"
print("[ok] 7. the logged reply starts at the injection, so pre-ref murmur is excluded")

print("\nAll search-state checks passed.")
