# Stability and answer quality

> **Before anything else, check which code the pod is running.**
> `grep "PIPELINE REVISION" live_server.log` must print the revision declared
> in `IMTalker/liveTry.py`. `conversation_logs_5` reported
> `ref_lora_loaded=False` while still injecting `<ref>` tags — a combination
> the current revision cannot produce, meaning the deployed server was older
> than the checkout being edited. The notebook now fails its health check on a
> revision mismatch.

## The reference LoRA — corrected

An earlier round defaulted `REF_LORA_ENABLED=0` on the assumption that this was
a third-party adapter leaking someone else's persona. **That was wrong.** It is
trained for this deployment, specifically so the model acts on injected
reference blocks. It is back ON by default.

The real defect was that it never loaded as trained:

```
reference LoRA coverage: 32/76 wrapped modules carry trained weights,
44 are zero-initialised no-ops
```

`scripts/download_live_assets.sh` publishes only `adapter_model.safetensors`
and hand-writes a config with **guessed suffix-style** `target_modules`
(`"proj"`, `"fc1"`, `"in_proj"`, …). PEFT treats those as suffixes and wraps
every module in the 7B whose name ends that way — far more than the checkpoint
trained. The extras get `lora_B = 0` (exact no-ops); the trained modules go
unmatched. An adapter applied to the wrong module set is not the adapter you
trained, which is why injected facts were ignored even with it enabled.

```bash
# derive a config that matches the checkpoint exactly
python stability/derive_ref_lora_config.py checkpoints/rag_lora/lora

# verify only (the launcher runs this on every start)
python stability/derive_ref_lora_config.py checkpoints/rag_lora/lora --check
```

It reads the module paths that actually carry weights and the rank off the
tensor shapes, then writes `target_modules` as **full paths** so PEFT wraps
exactly those. Startup should then log `reference LoRA fully applied (N/N)`.

`lora_alpha` cannot be recovered from weights. The tool defaults to `2 × r`
(matching the old 256/128). **If you trained with a different alpha, pass
`--alpha <value>`** — this is the one value only you know.

`REF_LORA_STRICT=1` (default) refuses to start on a partial load rather than
serving a half-applied adapter that looks healthy.


## Wrong answers: what `conversation_logs_1` showed

That session had four separate faults, all now fixed. If replies go wrong
again, this table is the triage order.

| Symptom in the log | Cause | Control |
| --- | --- | --- |
| "What is Bitcoin?" → *"is a virtual currency used on the platform… buying them in the shop"* | the `<lookup>/<ref>` adapter brought its training corpus's persona with it (r=128 / alpha=256 = scale 2.0 over the 4-bit base) | `REF_LORA_ENABLED=0` (new default), `REF_LORA_SCALE` |
| "who is the inventor of bitcoin" → *"Always the inventor of Big Ben"* | the STT decoded **11.8–35 s** per "utterance" (`stt_frames_decoded` 147–437), spanning the whole gap since the last turn | pre-roll ring + utterance gating + cap |
| a junk page summarised and injected as fact | relevance floor was `0.15`; the useful search scored 0.82, the useless ones 0.18–0.26 | `WEB_SEARCH_MIN_SCORE=0.50` |
| *"Today's guest list on the market has not been provided."* injected as grounding | the compressor's non-answers and question echoes were injected verbatim | new rejection gates |
| every reply logged missing its first words | the reply slice started 960 ms **after** the user stopped, but the model starts at +0.03 s | boundary moved to the user's first word |
| `<ref` and `coins.>` spoken aloud | tag stripping only matched complete tags | partial-tag + orphan-`>` patterns |

## `conversation_logs_2`: what the first round fixed, and what it exposed

Disabling the adapter worked — the unsearched turns became genuinely good
("You can buy Bitcoin directly on exchanges or through a broker…", "Bitcoin is
code, not a physical thing. It lives on a blockchain…"). Two faults were left
standing underneath it, both now fixed.

| Symptom | Cause | Fix |
| --- | --- | --- |
| **Every** `<ref>` injection produced a greeting instead of an answer — *"Okay, the S and P.> Thank you for calling RB Labs. Have a great day!"*, 4 turns out of 4 | the injection fed a **sine tone on the user stream**, which in this fork appears in exactly three places, all of them system-prompt priming. The model read the injection as "a new system prompt just ended" and did what it always does then: greet | user stream now carries **silence**; `IMTALKER_INJECT_USER_STREAM=sine` restores the old behaviour for A/B |
| *"What is Bitcoin?"* → *"A beat is a measure of timing"*; turn 1 → *"Hmm, I can't hear you well."* | **4.5 s of microphone audio discarded mid-question**. The backlog sits permanently near the 2.0 s cap (`frame_q_depth=32`, the backpressure limit), so the trim kept slicing into speech | trim now removes **silence only**, and cuts speech solely past a hard ceiling of 2.5× the cap |
| 3 of 4 searches injected *"There's no specific information available on this…"* | the fallback ref asserts, as a retrieved fact, that no facts exist | nothing is injected; the model answers from its own knowledge |

The transcript-window fix from the first round is confirmed working:
`stt_frames_decoded` fell from **147–437 frames (12–35 s)** to **23–56 frames
(1.8–4.5 s)**.

## `conversation_logs_3`: the injection fix landed, one regression found

Grounded answers came out correct for the first time — *"Today's gold price is
$140.72 per gram"*, *"Tesla's stock is valued at $341.64 per share"* — and the
un-searched turns were clean too (*"Bitcoin is a digital currency that runs on
blockchain. It's decentralized…"*). One fault remained, introduced by the
previous round.

| Symptom | Cause | Fix |
| --- | --- | --- |
| thinking sound looped **33 s** and never stopped, after tavily returned 0 results | the previous round cleared `search_awaiting_ref` **on the background thread** — the very flag that makes the GPU thread run `_consume_pending`. So the cancel path, which stops the clip and releases the text hold, was never reached | the background thread writes only `pending_search_cancelled`; the GPU thread owns every state transition |
| the next session answered *"What is Bitcoin?"* with *"big planes are huge for transporting lots of people"* | the looping clip echoed into the mic and was transcribed as *"It's Dolph. It's Dolph. It's Dolph…"* — that echo primed the fresh session's context, and it took two turns to wash out | fixed upstream by the hang; a watchdog now guarantees no clip outlives its search |
| *"Hmm, the stock market.> Today's gold price is…"* | a `>` left mid-reply where a tag's bracket survived the turn boundary, plus the model's full-duplex murmur while the question was still being asked | mid-string `.>` stripping; the logged reply now starts at the injection |

The watchdog is deliberately independent: it keys **only** on the clip being
active and sits at the top level of `_step`, so it fires on every 80 ms chunk
regardless of which flag is stuck. Budget is the filler cap + 2 s (8 s at
defaults) against the 33 s hang that was logged.

**Thread rule, now enforced by a test:** `_route_and_search` runs on a
background thread and may write **only** the `pending_*` handoff slots. Every
`search_*` state transition belongs to the GPU thread in `_consume_pending`.

## `conversation_logs_4`: the prompt was the last thing left

Everything upstream was finally clean — `total_dropped_speech_s=0.0`, correct
transcripts, correct search results, correct compression — and the assistant
still answered *"What is Bitcoin?"* with:

> *"I'd say it's because they're a small team with a big vision. They've been
> working on AI stuff for years, they're constantly fine tuning the models…"*

The system prompt is **force-fed as the assistant's own speech**. It ends
"…promote RB Labs robots when relevant", so the model believed it had just said
that and simply carried on. `--prompt_settle_sec` was `0`, leaving only 80 ms
between the prompt and the conversation. And because the KV cache is never
reset per turn, that monologue ran the entire session — it swallowed the next
question and both correctly retrieved facts (`$140.72/gram`, `$346.86`).

`conversation_logs_3` looked fine only because its first question was *"What is
your name?"*, which self-description happens to answer.

| Fix | Detail |
| --- | --- |
| `PROMPT_SETTLE_SEC` **0 → 0.96** | the same gap this system already treats as end-of-utterance (`_VAD_SILENCE_FRAMES_REQUIRED`) |
| settle pads in the **conversational** register | it used sine-on-user — the fork's *priming* marker — so padding kept the model **inside** the prompt. That is why long values used to mute it rather than free it. Silence-on-user is an ordinary gap, so the model reaches a turn boundary |
| `<ref>` / `<lookup>` tags only when the adapter is loaded | with `REF_LORA_ENABLED=0` the tags are untrained syntax the model reads aloud (*"Hmm, the stock market.>"*, *"coins.>"*, *"<ref"*). Plain text now goes in instead |

**Note on the prompt itself.** Because it arrives as the assistant's own
speech, every instruction reads as something it already said. "Give financial
advice" and "promote RB Labs robots" therefore bias it toward volunteering
advertising and advice instead of answering. That is a wording decision, not a
bug — but if replies still drift toward self-promotion, shorten
`IMTalker/prompts/RB_Robert_System_Prompt_full.txt` to identity and tone only.

**Start here if answers are wrong:** `REF_LORA_ENABLED=0` is now the default.
That single change is what stops the assistant answering finance questions in
the vocabulary of an app store. Injection still works without the adapter — a
`<ref>` block is force-fed into the live context either way.

```bash
# restore it, weakened, if you want its tag handling back
REF_LORA_ENABLED=1 REF_LORA_SCALE=0.3 bash run_live.sh
```

# Run-to-run stability

This directory exists because the same pod, the same notebook and the same
question could produce a correct answer on one run and unrelated speech on the
next. That was not one bug. It was several independent per-run draws stacked on
one pipeline, and nothing in the logs or the health check recorded which draw
you got.

## What changed

| Draw that used to vary per run | Fix | Where |
| --- | --- | --- |
| PersonaPlex sampled every token from an **unseeded** global RNG | seeded at engine init and on every session reset | `IMTalker/liveTry.py` |
| Sampling width was hard-coded and unreachable | `PERSONAPLEX_TEMP*` / `PERSONAPLEX_TOP_K*` | `IMTalker/liveTry.py`, `run_live.sh` |
| Wrong `moshi` fork loaded → 4-bit, CFG and the text hook dropped **silently** | strict runtime contract, refuses to start | `IMTalker/liveTry.py` |
| Flow-matching noise ignored the seeded buffer | `noise_init` now honoured | `IMTalker/generator/FM.py` |
| Stale `/workspace` checkpoints outranked the fresh checkout | checkout wins by default | `IMTalker/start_winner_live.sh` |
| Asset downloads pinned nothing | per-repo revision pinning + manifest | `scripts/download_live_assets.sh` |
| Two different repositories cloned by one notebook | one clone source, optional commit pin | the notebook |
| Dependency versions re-resolved on every pod | optional `requirements.lock.txt` applied last | the notebook, Step 6c |

`--seed 42` in `start_winner_live.sh` never reached PersonaPlex, and still
doesn't — it is read only by `generator/FM.py`, and only under
`--fix_noise_seed`. `PERSONAPLEX_SEED` is the flag that controls generation.

## Controls

All are read by `run_live.sh` and set from the notebook's Parameters cell.

| Variable | Default | Effect |
| --- | --- | --- |
| `PERSONAPLEX_SEED` | `42` | Seeds generation. Empty string = the old random behaviour. |
| `PERSONAPLEX_RESEED_PER_SESSION` | `1` | Re-seed on every WebSocket connect, so connection *n* matches connection 1. |
| `PERSONAPLEX_TEMP_TEXT` | *(build default)* | Text-stream temperature. **Lower this first** when replies drift off-topic — the text stream carries the meaning. |
| `PERSONAPLEX_TOP_K_TEXT` | *(build default)* | Text-stream top-k. Try `8` alongside `PERSONAPLEX_TEMP_TEXT=0.4`. |
| `PERSONAPLEX_TEMP` / `PERSONAPLEX_TOP_K` | *(build default)* | Audio-stream sampling. Mostly affects prosody. |

The four sampling variables are **empty by default on purpose**. An empty value
passes nothing to `LMGen`, so that build's own defaults survive — PersonaPlex
forks differ, and baking a number into the launcher would silently change
generation on a fork whose default is different. The startup line reports which
values are overrides and which are build defaults.
| `ALLOW_MOSHI_FALLBACK` | `0` | `0` refuses to start on a degraded runtime. `1` restores the old silent fallbacks. |
| `ALLOW_WORKSPACE_ASSET_FALLBACK` | `0` | `1` restores the old `/workspace/...`-first asset search. |

## Comparing a good run with a bad one

```bash
source /workspace/preprocess_5090/bin/activate
cd /workspace/speech2avatar

# On every pod (the notebook does this automatically at Step 13b):
python stability/run_fingerprint.py --out fingerprints/fp_$(date +%s).json

# When a pod misbehaves:
python stability/run_fingerprint.py --compare fp_good.json fp_bad.json
```

Differences that can change answer quality on their own are printed first and
marked `!!`. Exit code is non-zero when a critical or asset difference is found.

The fingerprint records which `moshi` file was imported and whether it silently
dropped anything, the sampling defaults in force, `torch.initial_seed()`, every
dependency version, and the sha256 of every checkpoint resolved through the
launcher's own precedence.

## The runtime contract

The server prints one line recording what it actually loaded, then
`PersonaPlex runtime contract OK` when nothing was lost:

```
[liveTry] runtime contract: moshi=/…/personaplex_bnb4/moshi/moshi/__init__.py v0.2.x
          lmgen=full dropped_loader_kwargs=[] condition_tensors=0(none-declared)
          seed=42 reseed_per_session=True sampling={'temp': 0.8, …}
[liveTry] PersonaPlex runtime contract OK
```

**A missing capability is only a fault when something is lost.** The gates are
built on that rule, because failing on mere absence just breaks healthy pods:

| `condition_source` | Meaning | Healthy? |
| --- | --- | --- |
| `built` | a `get_condition_tensors` helper produced them | yes |
| `none-declared` | the model declares no conditioners, so `{}` is correct | **yes** |
| `unsupported-api` | this LMGen takes no `condition_tensors` at all | yes |
| `missing-helper` | conditioners exist but nothing could build them | **no** |

`none-declared` is the normal result for the bnb4 fork: it ships no
`moshi.run_inference` module, and its model has no conditioners. Likewise a
`cfg_coef` the build cannot accept is fatal only when a scale other than `1.0`
was requested — the launcher leaves `--moshi_cfg_coef` at `1.0`, where CFG is a
no-op — and a dropped `quantize_4bit` is fatal only when `True` was passed.

`reference LoRA coverage: N/M wrapped modules carry trained weights` explains
PEFT's "missing adapter keys" warning: the hand-written `adapter_config.json`
targets more modules than the checkpoint populates. The unpopulated ones get
`lora_B = 0`, so they contribute exactly nothing — harmless, but now counted.

## Reading the server log

```bash
grep -n "PersonaPlex runtime contract OK" live_server.log   # must be present
grep -n "runtime contract:"              live_server.log   # the full detail line
grep -n "PersonaPlex sampling seeded"    live_server.log    # once per connect
grep -n -A14 "resolved assets"           live_server.log    # which files won
grep -c  "microphone backlog exceeded"   live_server.log    # >0 = truncated questions
grep -n  "component_status"              live_server.log    # what the search stack did
```

`conversation_logs/` has the per-turn story: `HEARD → DECIDE → SEARCH → GROUND
→ DONE → REPLIED`. If `HEARD` does not match what you said, the problem is the
microphone backlog or the STT, not PersonaPlex.

## Pinning a known-good pod

Once a pod answers correctly:

```bash
# 1. Freeze the environment (bitsandbytes above all -- it is the 4-bit kernel
#    for the whole 7B model on sm_120).
cp requirements.lock.suggested.txt requirements.lock.txt

# 2. Freeze the checkpoints.
cp checkpoints/asset_revisions.suggested.env asset_revisions.env

# 3. Freeze the code: set GIT_COMMIT in the notebook's Parameters cell to the
#    commit printed by Step 0.
```

Commit all three. Every later pod then resolves to the same software, the same
weights and the same source.

## Verifying the fixes

```bash
python stability/test_fixes.py .            # seeding, FM noise, wiring
python stability/test_runtime_contract.py   # condition tensors, gates, launchers
python stability/test_lmgen_call.py         # LMGen construction across forks
python stability/test_answer_quality.py     # every failure from conversation_logs_1
python stability/test_injection_and_audio.py # every failure from conversation_logs_2
python stability/test_search_state.py       # every failure from conversation_logs_3
python stability/test_prompt_boundary.py    # every failure from conversation_logs_4
python stability/test_ref_lora.py           # adapter load path + revision marker
```

Together they check that the seed helpers reproduce the sampling stream, that
`FM.sample` honours `noise_init` per chunk, that the strict gates fire only on
real losses, and that the `LMGen` call is built from the signature — including
the bnb4 fork, where `device` is a **required** positional and `cfg_coef` /
`condition_tensors` / `on_text_hook` do not exist. They need only `torch`.

## What is still nondeterministic

Bit-exact reproducibility is not achievable here and chasing it would cost the
real-time budget:

- **CUDA-graph replay and TF32.** Reduction order is not guaranteed identical
  across launches, so logits differ in the last bits. Under top-k sampling that
  is almost always invisible. Do not reach for
  `torch.use_deterministic_algorithms(True)` — it costs more latency than this
  pipeline has.
- **Live web results** when `ENABLE_SEARCH=1`. The same question at two
  different times is genuinely a different input.
- **Real-time scheduling.** How many 80 ms chunks land between the question
  ending and the reply starting depends on host load, and a full-duplex model
  is sensitive to that alignment by design.

The goal is that the remaining variance is *within* a run rather than *between*
pods — so a bad run can be reproduced, and a change can be measured.
