#!/bin/bash
# Full latent-analysis pipeline for the equinoise_libero_cosmos_full / steps_80000
# checkpoint. Regenerates checkpoint-specific caches+subspaces (tf experiment),
# then runs the 8-phase clean_action_latent_v2 analysis. Stops on first error.
set -uo pipefail

REPO=/workspace/starvla_latent
PY=/venv/starVLA/bin/python
export HF_HOME=/workspace/.hf_home
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

TF=$REPO/experiments/tf_latent_disentanglement
V2=$REPO/experiments/clean_action_latent_v2
TFCFG=$TF/configs/experiment_equinoise80k.yaml
V2CFG=$V2/configs/clean_action_latent_equinoise80k.yaml
LOG=$V2/outputs/equinoise_libero_cosmos_full_80k/_run_logs
mkdir -p "$LOG"
cd "$REPO"

run() {  # run <label> <logfile> <cmd...>
  local label="$1"; local logf="$2"; shift 2
  echo "==================================================================="
  echo "[$(date +%H:%M:%S)] START  $label"
  echo "    cmd: $*"
  "$@" > "$logf" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] FAILED $label  (rc=$rc)  -> tail of $logf:"
    tail -25 "$logf"
    echo "[ABORT] pipeline halted at: $label"
    exit $rc
  fi
  echo "[$(date +%H:%M:%S)] OK     $label"
}

echo "##### STAGE A: regenerate latents + subspaces (tf_latent_disentanglement) #####"
run "tf-01-extract"   "$LOG/tf01_extract.log"   $PY $TF/scripts/01_extract_latents.py     --config $TFCFG
run "tf-03-subspace"  "$LOG/tf03_subspace.log"  $PY $TF/scripts/03_subspace_decomposition.py --config $TFCFG
run "tf-04-probes"    "$LOG/tf04_probes.log"    $PY $TF/scripts/04_train_probes.py         --config $TFCFG

echo "##### STAGE B: clean_action_latent_v2 phases 0..8 #####"
run "v2-00-align"        "$LOG/v2_00.log" $PY $V2/scripts/action_latent/00_verify_probe_align.py   --config $V2CFG
run "v2-01-confound"     "$LOG/v2_01.log" $PY $V2/scripts/action_latent/01_confound_gate.py         --config $V2CFG
run "v2-02-linear"       "$LOG/v2_02.log" $PY $V2/scripts/action_latent/02_linear_action_subspace.py --config $V2CFG
run "v2-03-bottleneck"   "$LOG/v2_03.log" $PY $V2/scripts/action_latent/03_train_bottleneck.py      --config $V2CFG
run "v2-04-leakage"      "$LOG/v2_04.log" $PY $V2/scripts/action_latent/04_leakage_eval.py          --config $V2CFG
run "v2-05-retrieval"    "$LOG/v2_05.log" $PY $V2/scripts/action_latent/05_pair_retrieval.py        --config $V2CFG
run "v2-06-intervention" "$LOG/v2_06.log" $PY $V2/scripts/action_latent/06_intervention_eval.py     --config $V2CFG
run "v2-07-token"        "$LOG/v2_07.log" $PY $V2/scripts/action_latent/07_token_action.py          --config $V2CFG
run "v2-08-report"       "$LOG/v2_08.log" $PY $V2/scripts/action_latent/08_make_report.py           --config $V2CFG

echo "==================================================================="
echo "[$(date +%H:%M:%S)] PIPELINE COMPLETE"
echo "verdict:"; cat $V2/outputs/equinoise_libero_cosmos_full_80k/verdict.json 2>/dev/null | head -40
