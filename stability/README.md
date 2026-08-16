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
| `PERSONAPLEX_TEMP_TEXT` | `0.7` | Text-stream temperature. **Lower this first** when replies drift off-topic — the text stream carries the meaning. |
| `PERSONAPLEX_TOP_K_TEXT` | `25` | Text-stream top-k. Try `8` alongside `PERSONAPLEX_TEMP_TEXT=0.4`. |
| `PERSONAPLEX_TEMP` / `PERSONAPLEX_TOP_K` | `0.8` / `250` | Audio-stream sampling. Mostly affects prosody. |
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
python stability/test_fixes.py .
```

Checks the seed helpers make the sampling stream reproducible, that the strict
runtime gate accepts the PersonaPlex fork and refuses the repo fork, that
`FM.sample` honours `noise_init` per chunk, and that the launcher scripts carry
the contract. Needs only `torch`.

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
