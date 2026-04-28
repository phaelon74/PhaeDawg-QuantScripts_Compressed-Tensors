#!/usr/bin/env bash
#
# Batch KLD runner driven by Models_KLD.json (see Models_KLD.json.example).
# Requires: bash, jq, python3, du, hf (Hugging Face CLI), and a vLLM tree
# containing examples/offline_inference/score_mode_kld.py.
#
# Usage:
#   ./run_batch_kld.sh /path/to/Models_KLD.json
#   MODELS_KLD_JSON=/path/to/Models_KLD.json ./run_batch_kld.sh
#
# Example (Qwen3.6-35B-A3B BF16 reference logits vs listed quants; run from KLD_Scripts):
#   ./run_batch_kld.sh ./Models_KLD_Qwen3.6-35B-A3B.json
#
# Resume: if ${resultsBasename}_KLD-Results.json already exists under outputDirectory,
# any model with status "success" and a numeric meanKld is left unchanged; others
# are re-run. Set KLD_FORCE_RERUN=1 to always run score_mode_kld for every model.
# Set KLD_RESUME=0 (or false) to ignore the existing results file and re-run everyone.
# After a successful write, any non-base model with meanKld null is reported on stderr.
# Set KLD_FAIL_ON_INCOMPLETE_RESULTS=1 to exit 1 when that happens (CI-style).
#
# cleanupDownloadedModelAfterSuccess: only deletes a downloaded model dir after KLD
# succeeds (failed runs keep weights for a cheap resume). hf download is run under
# set +e; if it exits non-zero but the target dir has files, we continue to KLD.
#
# Before score_mode_kld: verify_hf_local_dir.py checks hub file sizes (+ sha256 when
# the Hub exposes it) and safetensors headers under --local-dir. On failure, hf
# download is retried up to HF_VERIFY_DOWNLOAD_RETRIES times (default 5). If set,
# HF_VERIFY_ATTEMPTS overrides that count (legacy name).
#
# Lessons from RunPod-style drivers (e.g. run_kld_benchmark.sh): run score_mode_kld
# through \`tee\` so logs hit disk and the terminal, and use \`PIPESTATUS[0]\` for the
# Python exit code (not \`$?\` after a pipe, which would be \`tee\`'s). Per-model work
# can be wrapped as \`func ... || log "continuing"\` so one failure never aborts the batch;
# here we use an explicit for-loop plus per-row JSON and optional KLD_FAIL_ON_INCOMPLETE_RESULTS.
# Set KLD_TEE_LOGS=0 to append only to the log file (no terminal copy).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_JSON="${1:-${MODELS_KLD_JSON:-}}"

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

[[ -n "$CONFIG_JSON" ]] || die "pass Models_KLD.json path as argv1 or set MODELS_KLD_JSON"
[[ -f "$CONFIG_JSON" ]] || die "config not found: $CONFIG_JSON"

need_cmd jq
need_cmd python3
need_cmd du
need_cmd hf

PARSE_PY="$SCRIPT_DIR/parse_mean_kld.py"
VERIFY_PY="$SCRIPT_DIR/verify_hf_local_dir.py"
[[ -f "$PARSE_PY" ]] || die "missing $PARSE_PY"
[[ -f "$VERIFY_PY" ]] || die "missing $VERIFY_PY"

CFG_ABS="$(cd "$(dirname "$CONFIG_JSON")" && pwd)/$(basename "$CONFIG_JSON")"

if ! _jq_cfg_err="$(jq empty "$CFG_ABS" 2>&1)"; then
  echo "error: config is not valid JSON: $CFG_ABS" >&2
  printf '%s\n' "$_jq_cfg_err" | sed 's/^/  /' >&2
  die "fix the JSON (common mistake: missing comma after the line above the one jq points to)."
fi
unset _jq_cfg_err

VLLM_ROOT="$(jq -r '.vllmRepositoryRoot' "$CFG_ABS")"
[[ -n "$VLLM_ROOT" && "$VLLM_ROOT" != "null" ]] || die "vllmRepositoryRoot missing in config"
VLLM_ROOT="${VLLM_ROOT%/}"

# scoreModeKldRelativePath: repo-relative (joined with vllmRepositoryRoot) or absolute path to score_mode_kld.py
SCORE_REL="$(jq -r '.scoreModeKldRelativePath // "examples/offline_inference/score_mode_kld.py"' "$CFG_ABS")"
if [[ "$SCORE_REL" == /* ]]; then
  SCORE_PY="$SCORE_REL"
else
  SCORE_PY="$VLLM_ROOT/$SCORE_REL"
fi
[[ -f "$SCORE_PY" ]] || die "score_mode_kld script not found: $SCORE_PY"

PROJECT="$(jq -r '.projectName // "KLD-Run"' "$CFG_ABS")"
RESULTS_BASENAME="$(jq -r '.resultsBasename // .projectName' "$CFG_ABS")"
OUT_DIR="$(jq -r '.outputDirectory // empty' "$CFG_ABS")"
if [[ -z "$OUT_DIR" || "$OUT_DIR" == "null" ]]; then
  OUT_DIR="$SCRIPT_DIR/output"
fi
OUT_DIR="${OUT_DIR%/}"
mkdir -p "$OUT_DIR/logs"

DOWNLOAD_IF_MISSING="$(jq -r '.downloadIfMissing // true' "$CFG_ABS")"
CLEANUP_DL="$(jq -r '.cleanupDownloadedModelAfterSuccess // true' "$CFG_ABS")"
RUN_KLD_BASE="$(jq -r '.runKldOnBaseModel // false' "$CFG_ABS")"
DATASET="$(jq -r '.dataset' "$CFG_ABS")"
DATASET_CFG="$(jq -r '.datasetConfig' "$CFG_ABS")"
TP="$(jq -r '.tensorParallelSize' "$CFG_ABS")"
GPU_MEM="$(jq -r '.gpuMemoryUtilization' "$CFG_ABS")"
DEFAULT_REF_LOGITS="$(jq -r '.defaultReferenceLogitsDir // empty' "$CFG_ABS")"

BASE_COUNT="$(jq '[.models[] | select(.isBase == true)] | length' "$CFG_ABS")"
if [[ "$BASE_COUNT" -gt 1 ]]; then
  die "config must contain at most one model with isBase:true"
fi
if [[ "$BASE_COUNT" -eq 1 ]]; then
  BASE_ID="$(jq -r '.models[] | select(.isBase == true) | .id' "$CFG_ABS")"
else
  BASE_ID=""
fi

RESULTS_JSON="$OUT_DIR/${RESULTS_BASENAME}_KLD-Results.json"
NDJSON_TMP="$(mktemp)"
MODELS_ARRAY_TMP="$(mktemp)"
trap 'rm -f "$NDJSON_TMP" "$MODELS_ARRAY_TMP"' EXIT

ISO_NOW="$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ")"

RESUME_OK=0
if [[ "${KLD_RESUME:-1}" =~ ^(0|false|no)$ ]]; then
  echo "resume: disabled (KLD_RESUME=${KLD_RESUME:-})." >&2
elif [[ -n "${KLD_FORCE_RERUN:-}" ]]; then
  echo "resume: disabled (KLD_FORCE_RERUN is set)." >&2
elif [[ -f "$RESULTS_JSON" ]] && jq empty "$RESULTS_JSON" >/dev/null 2>&1; then
  RESUME_OK=1
  echo "resume: using existing results at ${RESULTS_JSON} (models with numeric meanKld are skipped)." >&2
elif [[ -f "$RESULTS_JSON" ]]; then
  echo "warning: existing results file is not valid JSON; not resuming: ${RESULTS_JSON}" >&2
fi

# du|awk under bash pipefail returns non-zero if du errors (common on huge trees / NFS),
# which would kill the whole script (set -e) right after hf download.
disk_usage_human() {
  local p="$1" s
  if [[ ! -d "$p" ]]; then
    echo "0"
    return 0
  fi
  s="$(
    set +e
    set +o pipefail 2>/dev/null || true
    du -sh "$p" 2>/dev/null | awk '{print $1; exit}'
  )"
  if [[ -z "${s// /}" ]]; then
    echo "0"
  else
    printf '%s\n' "$s"
  fi
  return 0
}

disk_usage_kb() {
  local p="$1" s
  if [[ ! -d "$p" ]]; then
    echo "0"
    return 0
  fi
  s="$(
    set +e
    set +o pipefail 2>/dev/null || true
    du -sk "$p" 2>/dev/null | awk '{print $1; exit}'
  )"
  if [[ -z "${s// /}" ]]; then
    echo "0"
  else
    printf '%s\n' "$s"
  fi
  return 0
}

hf_download_to() {
  local hf_id="$1"
  local dest="$2"
  mkdir -p "$dest"
  set +e
  hf download "$hf_id" --local-dir "$dest"
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi
  if [[ -n "$(find "$dest" -mindepth 1 -maxdepth 8 -type f -print -quit 2>/dev/null)" ]]; then
    echo "warning: hf download exited ${rc} for ${hf_id}; ${dest} has files — continuing." >&2
    return 0
  fi
  die "hf download failed (exit ${rc}) and ${dest} looks empty: ${hf_id}"
}

# Retries hf download when verify_hf_local_dir.py fails (partial/corrupt snapshots).
ensure_hf_snapshot_matches_hub() {
  local hf_id="$1"
  local dest="$2"
  local mid="$3"
  local verify_log="$OUT_DIR/logs/${mid}_hf_verify.log"
  local retries="${HF_VERIFY_DOWNLOAD_RETRIES:-5}"
  [[ -n "${HF_VERIFY_ATTEMPTS:-}" ]] && retries="${HF_VERIFY_ATTEMPTS}"
  local n=0
  while true; do
    if python3 "$VERIFY_PY" --repo-id "$hf_id" --local-dir "$dest" >"$verify_log" 2>&1; then
      return 0
    fi
    if [[ "$n" -ge "$retries" ]]; then
      echo "HF verify still failing after ${retries} repair download(s); tail of ${verify_log}:" >&2
      tail -n 50 "$verify_log" >&2 || true
      return 1
    fi
    echo "HF verify failed; re-running hf download for ${hf_id} (repair $((n + 1))/${retries})..." >&2
    hf_download_to "$hf_id" "$dest"
    REDOWNLOADED_FOR_VERIFY=1
    n=$((n + 1))
  done
}

run_one_kld() {
  local model_path="$1"
  local ref_mode="$2"
  local ref_logit="$3"
  local ref_model="$4"
  local log_file="$5"

  : >"$log_file"
  if [[ "$ref_mode" == "logits" ]]; then
    [[ -n "$ref_logit" && "$ref_logit" != "null" ]] || die "reference logits path required for logits mode"
    [[ -d "$ref_logit" ]] || die "reference logits directory missing: $ref_logit"
    if [[ "${KLD_TEE_LOGS:-1}" =~ ^(1|true|yes)$ ]]; then
      python3 "$SCORE_PY" \
        --model "$model_path" \
        --reference-logits "$ref_logit" \
        --dataset "$DATASET" \
        --dataset-config "$DATASET_CFG" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization "$GPU_MEM" \
        2>&1 | tee "$log_file"
      return "${PIPESTATUS[0]}"
    fi
    python3 "$SCORE_PY" \
      --model "$model_path" \
      --reference-logits "$ref_logit" \
      --dataset "$DATASET" \
      --dataset-config "$DATASET_CFG" \
      --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$GPU_MEM" \
      >>"$log_file" 2>&1
    return $?
  else
    [[ -n "$ref_model" && "$ref_model" != "null" ]] || die "reference model path required for model mode"
    [[ -d "$ref_model" ]] || die "reference model directory missing: $ref_model"
    if [[ "${KLD_TEE_LOGS:-1}" =~ ^(1|true|yes)$ ]]; then
      python3 "$SCORE_PY" \
        --model "$model_path" \
        --reference-model "$ref_model" \
        --dataset "$DATASET" \
        --dataset-config "$DATASET_CFG" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization "$GPU_MEM" \
        2>&1 | tee "$log_file"
      return "${PIPESTATUS[0]}"
    fi
    python3 "$SCORE_PY" \
      --model "$model_path" \
      --reference-model "$ref_model" \
      --dataset "$DATASET" \
      --dataset-config "$DATASET_CFG" \
      --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$GPU_MEM" \
      >>"$log_file" 2>&1
    return $?
  fi
}

MODEL_LEN="$(jq '.models | length' "$CFG_ABS")"
for ((i = 0; i < MODEL_LEN; i++)); do
  MID="$(jq -r ".models[$i].id" "$CFG_ABS")"
  IS_BASE="$(jq -r ".models[$i].isBase" "$CFG_ABS")"
  HF_ID="$(jq -r ".models[$i].huggingfaceId" "$CFG_ABS")"
  LOCAL="$(jq -r ".models[$i].localPath" "$CFG_ABS")"
  REF_MODE="$(jq -r '.models['"$i"'].referenceMode // "logits"' "$CFG_ABS")"
  REF_LOGIT="$(jq -r ".models[$i].referenceLogitsPath // empty" "$CFG_ABS")"
  REF_MODEL="$(jq -r ".models[$i].referenceModelPath // empty" "$CFG_ABS")"
  MEAN_KLD=""
  STATUS="pending"
  DETAIL=""
  DOWNLOADED=0
  REDOWNLOADED_FOR_VERIFY=0

  if [[ -z "$REF_LOGIT" || "$REF_LOGIT" == "null" ]]; then
    REF_LOGIT="$DEFAULT_REF_LOGITS"
  fi
  if [[ -z "$REF_MODEL" || "$REF_MODEL" == "null" ]]; then
    REF_MODEL=""
  fi

  LOG_FILE="$OUT_DIR/logs/${MID}_kld.log"

  if [[ "$RESUME_OK" -eq 1 ]]; then
    # Single jq stream (no "| head"). Re-validate in bash so we never skip on a bogus line.
    CACHED_ROW="$(jq -c --arg id "$MID" '
      first(
        .models[]?
        | select(.id == $id)
        | select(.status == "success")
        | select((.meanKld | type) == "number")
      )
    ' "$RESULTS_JSON" 2>/dev/null)" || true
    if [[ -n "$CACHED_ROW" && "${CACHED_ROW:0:1}" == "{" ]] \
      && echo "$CACHED_ROW" | jq -e --arg id "$MID" \
        '(.id == $id) and (.status == "success") and ((.meanKld | type) == "number")' >/dev/null 2>&1; then
      printf '%s\n' "$CACHED_ROW" >>"$NDJSON_TMP"
      echo "resume: skipping KLD for ${MID} (meanKld already present)." >&2
      continue
    fi
  fi

  echo "processing: ${MID}" >&2

  if [[ "$IS_BASE" == "true" && "$RUN_KLD_BASE" != "true" && ! -d "$LOCAL" && "$DOWNLOAD_IF_MISSING" == "true" ]]; then
    hf_download_to "$HF_ID" "$LOCAL"
  fi

  if [[ "$IS_BASE" == "true" && "$RUN_KLD_BASE" != "true" ]]; then
    DU_H="$(disk_usage_human "$LOCAL")"
    DU_K="$(disk_usage_kb "$LOCAL")"
    jq -nc \
      --arg id "$MID" \
      --argjson is_base true \
      --arg hf "$HF_ID" \
      --arg lp "$LOCAL" \
      --arg duh "$DU_H" \
      --arg duk "$DU_K" \
      --arg st "skipped" \
      --arg det "Base model; runKldOnBaseModel is false in config." \
      '{id:$id,isBase:$is_base,huggingfaceId:$hf,localPath:$lp,diskUsageHuman:$duh,diskUsageKilobytes:($duk|tonumber),meanKld:null,status:$st,detail:$det,kldLogPath:null}' >>"$NDJSON_TMP"
    continue
  fi

  if [[ ! -d "$LOCAL" ]]; then
    if [[ "$DOWNLOAD_IF_MISSING" != "true" ]]; then
      DU_H="$(disk_usage_human "$LOCAL")"
      DU_K="$(disk_usage_kb "$LOCAL")"
      jq -nc \
        --arg id "$MID" \
        --argjson is_base "$( [[ "$IS_BASE" == "true" ]] && echo true || echo false )" \
        --arg hf "$HF_ID" \
        --arg lp "$LOCAL" \
        --arg duh "$DU_H" \
        --arg duk "$DU_K" \
        --arg st "failed" \
        --arg det "localPath missing and downloadIfMissing is false" \
        '{id:$id,isBase:$is_base,huggingfaceId:$hf,localPath:$lp,diskUsageHuman:$duh,diskUsageKilobytes:($duk|tonumber),meanKld:null,status:$st,detail:$det,kldLogPath:null}' >>"$NDJSON_TMP"
      continue
    fi
    hf_download_to "$HF_ID" "$LOCAL"
    DOWNLOADED=1
  fi

  VERIFY_LOG="$OUT_DIR/logs/${MID}_hf_verify.log"
  if ! ensure_hf_snapshot_matches_hub "$HF_ID" "$LOCAL" "$MID"; then
    DU_H="$(disk_usage_human "$LOCAL")"
    DU_K="$(disk_usage_kb "$LOCAL")"
    IS_BASE_JSON="false"
    [[ "$IS_BASE" == "true" ]] && IS_BASE_JSON="true"
    jq -nc \
      --arg id "$MID" \
      --argjson is_base "$IS_BASE_JSON" \
      --arg hf "$HF_ID" \
      --arg lp "$LOCAL" \
      --arg duh "$DU_H" \
      --arg duk "$DU_K" \
      --arg st "failed" \
      --arg det "HF snapshot verification failed vs hub (sizes/sha256/safetensors); see ${VERIFY_LOG}" \
      --arg vlp "$VERIFY_LOG" \
      '{id:$id,isBase:$is_base,huggingfaceId:$hf,localPath:$lp,diskUsageHuman:$duh,diskUsageKilobytes:($duk|tonumber),meanKld:null,status:$st,detail:$det,kldLogPath:$vlp}' >>"$NDJSON_TMP"
    continue
  fi
  if [[ "${REDOWNLOADED_FOR_VERIFY:-0}" -eq 1 ]]; then
    DOWNLOADED=1
  fi

  DU_H="$(disk_usage_human "$LOCAL")"
  DU_K="$(disk_usage_kb "$LOCAL")"

  set +e
  run_one_kld "$LOCAL" "$REF_MODE" "$REF_LOGIT" "$REF_MODEL" "$LOG_FILE"
  KLD_RC=$?
  set -e

  # score_mode_kld often prints "Mean KLD:" then exits non-zero (e.g. NCCL / distributed
  # teardown warnings). Always parse the log; treat as success when Mean KLD is present.
  MK="$(python3 "$PARSE_PY" "$LOG_FILE" 2>/dev/null || true)"
  if [[ -n "$MK" ]]; then
    STATUS="success"
    MEAN_KLD="$MK"
    if [[ "$KLD_RC" -ne 0 ]]; then
      DETAIL="score_mode_kld.py exited with code ${KLD_RC} but Mean KLD was parsed from ${LOG_FILE}."
    else
      DETAIL=""
    fi
  elif [[ "$KLD_RC" -ne 0 ]]; then
    STATUS="failed"
    DETAIL="score_mode_kld.py exited with code $KLD_RC (see log)."
    MEAN_KLD=""
  else
    STATUS="failed"
    DETAIL="KLD run finished (exit 0) but mean KLD could not be parsed; inspect log."
    MEAN_KLD=""
  fi

  IS_BASE_JSON="false"
  [[ "$IS_BASE" == "true" ]] && IS_BASE_JSON="true"

  if [[ "$STATUS" == "success" && -n "$MEAN_KLD" ]]; then
    jq -nc \
      --arg id "$MID" \
      --argjson is_base "$IS_BASE_JSON" \
      --arg hf "$HF_ID" \
      --arg lp "$LOCAL" \
      --arg duh "$DU_H" \
      --arg duk "$DU_K" \
      --arg mk "$MEAN_KLD" \
      --arg st "$STATUS" \
      --arg det "$DETAIL" \
      --arg klp "$LOG_FILE" \
      '{id:$id,isBase:$is_base,huggingfaceId:$hf,localPath:$lp,diskUsageHuman:$duh,diskUsageKilobytes:($duk|tonumber),meanKld:($mk|tonumber),status:$st,detail:(if $det=="" then null else $det end),kldLogPath:$klp}' >>"$NDJSON_TMP"
  else
    jq -nc \
      --arg id "$MID" \
      --argjson is_base "$IS_BASE_JSON" \
      --arg hf "$HF_ID" \
      --arg lp "$LOCAL" \
      --arg duh "$DU_H" \
      --arg duk "$DU_K" \
      --arg st "$STATUS" \
      --arg det "$DETAIL" \
      --arg klp "$LOG_FILE" \
      '{id:$id,isBase:$is_base,huggingfaceId:$hf,localPath:$lp,diskUsageHuman:$duh,diskUsageKilobytes:($duk|tonumber),meanKld:null,status:$st,detail:$det,kldLogPath:$klp}' >>"$NDJSON_TMP"
  fi

  # cleanupDownloadedModelAfterSuccess: only remove weights we fetched in this run after a successful KLD
  # (on failure, keep the tree so a resume can retry without another multi-GB download).
  if [[ "$DOWNLOADED" -eq 1 && "$CLEANUP_DL" == "true" && "$IS_BASE" != "true" && "$STATUS" == "success" ]]; then
    rm -rf "$LOCAL"
  elif [[ "$DOWNLOADED" -eq 1 && "$CLEANUP_DL" == "true" && "$IS_BASE" == "true" ]]; then
    echo "note: model $MID was downloaded and is marked base; not auto-deleting." >&2
  fi
done

ND_LINES="$(wc -l <"$NDJSON_TMP" | tr -d ' \t')"
if [[ "$ND_LINES" -ne "$MODEL_LEN" ]]; then
  die "internal error: expected ${MODEL_LEN} per-model result lines, got ${ND_LINES} (see stderr above for each 'processing:' line)."
fi

jq -s '.' "$NDJSON_TMP" >"$MODELS_ARRAY_TMP"

jq -n \
  --arg ver "1" \
  --arg pn "$PROJECT" \
  --arg rb "$RESULTS_BASENAME" \
  --arg ga "$ISO_NOW" \
  --arg cp "$CFG_ABS" \
  --arg rp "$RESULTS_JSON" \
  --arg bid "${BASE_ID}" \
  --arg vr "$VLLM_ROOT" \
  --arg ds "$DATASET" \
  --arg dc "$DATASET_CFG" \
  --slurpfile models "$MODELS_ARRAY_TMP" \
  '{
    schemaVersion: ($ver|tonumber),
    projectName:$pn,
    resultsBasename:$rb,
    generatedAt:$ga,
    configPath:$cp,
    resultsPath:$rp,
    baseModelId:(if $bid=="" then null else $bid end),
    vllmRepositoryRoot:$vr,
    dataset:$ds,
    datasetConfig:$dc,
    models:$models[0]
  }' >"$RESULTS_JSON"

NULL_IDS="$(jq -r '[.models[] | select(.isBase == false) | select(.meanKld == null) | .id] | join(", ")' "$RESULTS_JSON")"
if [[ -n "$NULL_IDS" ]]; then
  echo "warning: non-base models still have meanKld null: ${NULL_IDS}" >&2
  echo "  (base stays null when runKldOnBaseModel is false; per-model failures do not stop the batch.)" >&2
  if [[ "${KLD_FAIL_ON_INCOMPLETE_RESULTS:-}" =~ ^(1|true|yes)$ ]]; then
    exit 1
  fi
fi

echo ""
echo "KLD batch finished."
echo "Results JSON: $RESULTS_JSON"
echo ""
