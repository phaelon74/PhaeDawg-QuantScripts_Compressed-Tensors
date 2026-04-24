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
[[ -f "$PARSE_PY" ]] || die "missing $PARSE_PY"

CFG_ABS="$(cd "$(dirname "$CONFIG_JSON")" && pwd)/$(basename "$CONFIG_JSON")"

VLLM_ROOT="$(jq -r '.vllmRepositoryRoot' "$CFG_ABS")"
[[ -n "$VLLM_ROOT" && "$VLLM_ROOT" != "null" ]] || die "vllmRepositoryRoot missing in config"
SCORE_REL="$(jq -r '.scoreModeKldRelativePath // "examples/offline_inference/score_mode_kld.py"' "$CFG_ABS")"
SCORE_PY="$VLLM_ROOT/$SCORE_REL"
[[ -f "$SCORE_PY" ]] || die "score_mode_kld script not found: $SCORE_PY"

PROJECT="$(jq -r '.projectName // "KLD-Run"' "$CFG_ABS")"
RESULTS_BASENAME="$(jq -r '.resultsBasename // .projectName' "$CFG_ABS")"
OUT_DIR="$(jq -r '.outputDirectory // empty' "$CFG_ABS")"
if [[ -z "$OUT_DIR" || "$OUT_DIR" == "null" ]]; then
  OUT_DIR="$SCRIPT_DIR/output"
fi
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

disk_usage_human() {
  local p="$1"
  if [[ -d "$p" ]]; then
    du -sh "$p" 2>/dev/null | awk '{print $1}'
  else
    echo "0"
  fi
}

disk_usage_kb() {
  local p="$1"
  if [[ -d "$p" ]]; then
    du -sk "$p" 2>/dev/null | awk '{print $1}'
  else
    echo "0"
  fi
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
    python3 "$SCORE_PY" \
      --model "$model_path" \
      --reference-logits "$ref_logit" \
      --dataset "$DATASET" \
      --dataset-config "$DATASET_CFG" \
      --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$GPU_MEM" \
      >>"$log_file" 2>&1
  else
    [[ -n "$ref_model" && "$ref_model" != "null" ]] || die "reference model path required for model mode"
    [[ -d "$ref_model" ]] || die "reference model directory missing: $ref_model"
    python3 "$SCORE_PY" \
      --model "$model_path" \
      --reference-model "$ref_model" \
      --dataset "$DATASET" \
      --dataset-config "$DATASET_CFG" \
      --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$GPU_MEM" \
      >>"$log_file" 2>&1
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

  if [[ -z "$REF_LOGIT" || "$REF_LOGIT" == "null" ]]; then
    REF_LOGIT="$DEFAULT_REF_LOGITS"
  fi
  if [[ -z "$REF_MODEL" || "$REF_MODEL" == "null" ]]; then
    REF_MODEL=""
  fi

  LOG_FILE="$OUT_DIR/logs/${MID}_kld.log"

  if [[ "$IS_BASE" == "true" && "$RUN_KLD_BASE" != "true" && ! -d "$LOCAL" && "$DOWNLOAD_IF_MISSING" == "true" ]]; then
    mkdir -p "$LOCAL"
    hf download "$HF_ID" --local-dir "$LOCAL"
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
    mkdir -p "$LOCAL"
    hf download "$HF_ID" --local-dir "$LOCAL"
    DOWNLOADED=1
  fi

  DU_H="$(disk_usage_human "$LOCAL")"
  DU_K="$(disk_usage_kb "$LOCAL")"

  set +e
  run_one_kld "$LOCAL" "$REF_MODE" "$REF_LOGIT" "$REF_MODEL" "$LOG_FILE"
  KLD_RC=$?
  set -e

  if [[ "$KLD_RC" -ne 0 ]]; then
    STATUS="failed"
    DETAIL="score_mode_kld.py exited with code $KLD_RC (see log)."
    MEAN_KLD=""
  else
    MK="$(python3 "$PARSE_PY" "$LOG_FILE" || true)"
    if [[ -z "$MK" ]]; then
      STATUS="failed"
      DETAIL="KLD run finished but mean KLD could not be parsed; inspect log."
      MEAN_KLD=""
    else
      STATUS="success"
      DETAIL=""
      MEAN_KLD="$MK"
    fi
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
      --arg klp "$LOG_FILE" \
      '{id:$id,isBase:$is_base,huggingfaceId:$hf,localPath:$lp,diskUsageHuman:$duh,diskUsageKilobytes:($duk|tonumber),meanKld:($mk|tonumber),status:$st,detail:null,kldLogPath:$klp}' >>"$NDJSON_TMP"
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

  if [[ "$DOWNLOADED" -eq 1 && "$CLEANUP_DL" == "true" && "$IS_BASE" != "true" ]]; then
    rm -rf "$LOCAL"
  elif [[ "$DOWNLOADED" -eq 1 && "$CLEANUP_DL" == "true" && "$IS_BASE" == "true" ]]; then
    echo "note: model $MID was downloaded and is marked base; not auto-deleting." >&2
  fi
done

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

echo ""
echo "KLD batch finished."
echo "Results JSON: $RESULTS_JSON"
echo ""
