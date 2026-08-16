#!/usr/bin/env bash
set -euo pipefail

ROOT="${SPEECH2AVATAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMTALKER_DIR="${IMTALKER_DIR:-$ROOT/IMTalker}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
PERSONAPLEX_BNB4_DIR="${PERSONAPLEX_BNB4_DIR:-$ROOT/checkpoints/personaplex_bnb4}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "$IMTALKER_DIR/checkpoints" "$CHECKPOINT_DIR" "$PERSONAPLEX_BNB4_DIR"

# --- Revision pinning -------------------------------------------------------
# None of these downloads used to pin a revision, so two pods built on different
# days could fetch different weights for the same "same repo, same notebook"
# run. Put the commit SHAs in asset_revisions.env (next to this repo's root) and
# every download below resolves to exactly those commits. The file is written
# for you the first time you run this script -- see ASSET_MANIFEST below -- so
# the workflow is: run once on a pod that behaves correctly, copy the resolved
# revisions into asset_revisions.env, commit it, and every later pod is pinned.
REVISIONS_FILE="${REVISIONS_FILE:-$ROOT/asset_revisions.env}"
if [[ -f "$REVISIONS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$REVISIONS_FILE"
  echo "[pin] using revisions from $REVISIONS_FILE"
else
  echo "[pin] no $REVISIONS_FILE -- downloading repo HEADs (unpinned)."
  echo "      Pin them after a known-good run: see $CHECKPOINT_DIR/ASSET_MANIFEST.txt"
fi

REV_IMTALKER="${REV_IMTALKER:-}"
REV_HDTF="${REV_HDTF:-}"
REV_UNITALK="${REV_UNITALK:-}"
REV_PERSONAPLEX_BNB4="${REV_PERSONAPLEX_BNB4:-}"
REV_PERSONAPLEX_NVIDIA="${REV_PERSONAPLEX_NVIDIA:-}"
REV_REF_LORA="${REV_REF_LORA:-}"

# hf download wrapper: appends --revision only when that repo has a pin.
# Usage: hf_get <revision-var-value> <repo> [args...]
hf_get() {
  local rev="$1"; shift
  if [[ -n "$rev" ]]; then
    hf download --revision "$rev" "$@"
  else
    hf download "$@"
  fi
}

echo "[1/10] Downloading IMTalker pretrained renderer/generator/wav2vec files..."
for f in \
  config.yaml \
  renderer.ckpt \
  generator.ckpt \
  wav2vec2-base-960h/config.json \
  wav2vec2-base-960h/pytorch_model.bin \
  wav2vec2-base-960h/preprocessor_config.json \
  wav2vec2-base-960h/feature_extractor_config.json
do
  hf_get "$REV_IMTALKER" cbsjtu01/IMTalker "$f" --local-dir "$IMTALKER_DIR/checkpoints"
done

echo "[2/10] Downloading PersonaPlex Helium->Wav2Vec adapter checkpoint..."
hf_get "$REV_HDTF" niloy629/hdtf_preprocess \
  personaplex_helium_w2v_frontend_adapter/checkpoints/phase2_best_wav2vec_final_loss.pt \
  --repo-type dataset \
  --local-dir "$CHECKPOINT_DIR"

echo "[3/10] Downloading UniTalk Phase2 latest epoch-4 adapter..."
hf_get "$REV_UNITALK" asifcsai/UniTalk \
  adapters/moshi_to_adapter/adapter_phase2_latest_ep4.pt \
  --local-dir "$CHECKPOINT_DIR/unitalk"

echo "[4/10] Downloading current look-ahead RMS adapter..."
hf_get "$REV_HDTF" niloy629/hdtf_preprocess \
  personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt \
  --repo-type dataset \
  --local-dir "$CHECKPOINT_DIR"

echo "[5/10] Downloading real-silence Helium deque seed..."
hf_get "$REV_HDTF" niloy629/hdtf_preprocess \
  personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt \
  --repo-type dataset \
  --local-dir "$CHECKPOINT_DIR"

echo "[6/10] Downloading current 2-hour static-head live LoRA checkpoint..."
hf_get "$REV_HDTF" niloy629/hdtf_preprocess \
  live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt \
  --repo-type dataset \
  --local-dir "$CHECKPOINT_DIR"

echo "[7/10] Downloading cached blink motion..."
hf_get "$REV_HDTF" niloy629/hdtf_preprocess \
  lora/3robert_audio3_ditto_static_motion.pt \
  --repo-type dataset \
  --local-dir "$CHECKPOINT_DIR"

echo "[8/10] Downloading PersonaPlex bnb4 weights..."
hf_get "$REV_PERSONAPLEX_BNB4" brianmatzelle/personaplex-7b-v1-bnb-4bit \
  --local-dir "$PERSONAPLEX_BNB4_DIR"

echo "[9/10] Downloading gated PersonaPlex Mimi/tokenizer assets..."
hf_get "$REV_PERSONAPLEX_NVIDIA" nvidia/personaplex-7b-v1 \
  tokenizer-e351c8d8-checkpoint125.safetensors \
  tokenizer_spm_32k_3.model \
  --local-dir "$PERSONAPLEX_BNB4_DIR"

echo "[10/11] Downloading PersonaPlex voices (VARM3 is the live default)..."
hf_get "$REV_PERSONAPLEX_NVIDIA" nvidia/personaplex-7b-v1 voices.tgz --local-dir "$PERSONAPLEX_BNB4_DIR"
tar -xzf "$PERSONAPLEX_BNB4_DIR/voices.tgz" -C "$PERSONAPLEX_BNB4_DIR"

# --- Reference-LoRA addition -------------------------------------------------
# Optional: only needed when the live server is launched with ENABLE_SEARCH=1.
# This adapter teaches the model to consume the injected <lookup>/<ref> tags;
# it is unrelated to where the referenced text came from, so it is still
# required now that the text comes from a web search. Never fails the whole
# asset pass if the repo is unreachable — search is additive, the avatar must
# still be able to boot without it.
REF_LORA_DIR="${REF_LORA_DIR:-${RAG_LORA_DIR:-$CHECKPOINT_DIR/rag_lora}}"
echo "[11/11] Downloading reference LoRA adapter (Darknsu/helium_lora_v1)..."
if mkdir -p "$REF_LORA_DIR/lora" && hf_get "$REV_REF_LORA" Darknsu/helium_lora_v1 \
  adapter_model.safetensors \
  --repo-type dataset \
  --local-dir "$REF_LORA_DIR/lora"
then
  # adapter_config.json is not published in that dataset repo (only the
  # weights are) -- write the matching config by hand. This is the exact
  # config the adapter was trained/saved with; if the adapter is ever
  # retrained, update this block (or add adapter_config.json to the HF repo).
  cat > "$REF_LORA_DIR/lora/adapter_config.json" <<'JSON'
{
  "alora_invocation_tokens": null,
  "alpha_pattern": {},
  "arrow_config": null,
  "auto_mapping": null,
  "base_model_name_or_path": null,
  "bias": "none",
  "corda_config": null,
  "ensure_weight_tying": false,
  "eva_config": null,
  "exclude_modules": null,
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layer_replication": null,
  "layers_pattern": null,
  "layers_to_transform": null,
  "loftq_config": {},
  "lora_alpha": 256.0,
  "lora_bias": false,
  "lora_dropout": 0.05,
  "lora_ga_config": null,
  "megatron_config": null,
  "megatron_core": "megatron.core",
  "modules_to_save": null,
  "peft_type": "LORA",
  "peft_version": "0.19.1",
  "qalora_group_size": 16,
  "r": 128,
  "rank_pattern": {},
  "revision": null,
  "target_modules": ["proj", "fc1", "out_proj", "fc2", "linear", "in_proj"],
  "target_parameters": null,
  "task_type": "FEATURE_EXTRACTION",
  "trainable_token_indices": null,
  "use_bdlora": null,
  "use_dora": false,
  "use_qalora": false,
  "use_rslora": false
}
JSON
  echo "  reference LoRA ready: $REF_LORA_DIR/lora"
else
  echo "  [warn] reference LoRA download failed/unreachable -- continuing without it." >&2
  echo "         Re-run this script later, or launch with ENABLE_SEARCH=0." >&2
fi

# --- Asset manifest ---------------------------------------------------------
# Records the commit each repo resolved to and the sha256 of every runtime file,
# so a good pod and a bad pod can be diffed directly, and so the revisions can
# be copied into asset_revisions.env to pin them permanently. Never fails the
# download pass: this is a record, not a gate.
MANIFEST="$CHECKPOINT_DIR/ASSET_MANIFEST.txt"
REV_SUGGEST="$CHECKPOINT_DIR/asset_revisions.suggested.env"
echo "Writing asset manifest to $MANIFEST"
{
  echo "# generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# repo commits actually downloaded (copy into asset_revisions.env to pin)"
} > "$MANIFEST"

python - "$MANIFEST" "$REV_SUGGEST" <<'PYEOF' || echo "  [warn] could not resolve repo revisions (recorded checksums only)" >&2
import sys
from huggingface_hub import HfApi

manifest_path, suggest_path = sys.argv[1], sys.argv[2]
repos = [
    ("REV_IMTALKER", "cbsjtu01/IMTalker", "model"),
    ("REV_HDTF", "niloy629/hdtf_preprocess", "dataset"),
    ("REV_UNITALK", "asifcsai/UniTalk", "model"),
    ("REV_PERSONAPLEX_BNB4", "brianmatzelle/personaplex-7b-v1-bnb-4bit", "model"),
    ("REV_PERSONAPLEX_NVIDIA", "nvidia/personaplex-7b-v1", "model"),
    ("REV_REF_LORA", "Darknsu/helium_lora_v1", "dataset"),
]
api = HfApi()
lines, env_lines = [], ["# Pin every asset download to these commits.",
                        "# cp checkpoints/asset_revisions.suggested.env asset_revisions.env"]
for var, repo, kind in repos:
    try:
        sha = api.repo_info(repo, repo_type=kind).sha
    except Exception as e:
        sha = f"<unresolved: {type(e).__name__}>"
    lines.append(f"{var}={sha}  # {repo} ({kind})")
    env_lines.append(f'{var}="{sha}"')
with open(manifest_path, "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(suggest_path, "w", encoding="utf-8") as f:
    f.write("\n".join(env_lines) + "\n")
print("  resolved " + str(len(repos)) + " repo revisions")
PYEOF

{
  echo
  echo "# sha256 of every file the live server loads"
} >> "$MANIFEST"
for f in \
  "$IMTALKER_DIR/checkpoints/generator.ckpt" \
  "$IMTALKER_DIR/checkpoints/renderer.ckpt" \
  "$IMTALKER_DIR/checkpoints/wav2vec2-base-960h/pytorch_model.bin" \
  "$PERSONAPLEX_BNB4_DIR/model_bnb_4bit.pt" \
  "$PERSONAPLEX_BNB4_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
  "$PERSONAPLEX_BNB4_DIR/tokenizer_spm_32k_3.model" \
  "$PERSONAPLEX_BNB4_DIR/voices/VARM3.pt" \
  "$CHECKPOINT_DIR/personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt" \
  "$CHECKPOINT_DIR/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt" \
  "$CHECKPOINT_DIR/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt" \
  "$CHECKPOINT_DIR/lora/3robert_audio3_ditto_static_motion.pt" \
  "$REF_LORA_DIR/lora/adapter_model.safetensors"; do
  if [[ -f "$f" ]]; then
    sha256sum "$f" >> "$MANIFEST"
  else
    echo "MISSING  $f" >> "$MANIFEST"
  fi
done
echo "  manifest: $MANIFEST"
echo "  suggested pins: $REV_SUGGEST"

echo
echo "Done. Key paths:"
echo "  renderer: $IMTALKER_DIR/checkpoints/renderer.ckpt"
echo "  generator: $IMTALKER_DIR/checkpoints/generator.ckpt"
echo "  wav2vec: $IMTALKER_DIR/checkpoints/wav2vec2-base-960h"
echo "  adapter: $CHECKPOINT_DIR/personaplex_helium_w2v_frontend_adapter/checkpoints/phase2_best_wav2vec_final_loss.pt"
echo "  UniTalk adapter: $CHECKPOINT_DIR/unitalk/adapters/moshi_to_adapter/adapter_phase2_latest_ep4.pt"
echo "  look-ahead RMS adapter: $CHECKPOINT_DIR/personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt"
echo "  silence Helium seed: $CHECKPOINT_DIR/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt"
echo "  live lora: $CHECKPOINT_DIR/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt"
echo "  blink motion: $CHECKPOINT_DIR/lora/3robert_audio3_ditto_static_motion.pt"
echo "  PersonaPlex bnb4: $PERSONAPLEX_BNB4_DIR"
echo "  default voice: $PERSONAPLEX_BNB4_DIR/voices/VARM3.pt"
echo "  reference LoRA (optional): $REF_LORA_DIR/lora"
