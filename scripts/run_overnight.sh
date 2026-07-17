#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Overnight experiment queue (MSc thesis: HAM10000 + ISIC 2020).
# Runs each job sequentially, logs everything, CONTINUES past failures,
# and writes a human-readable summary. Safe to leave running unattended.
#
# Order = value-first: HAM baselines + engineered metadata + the
# leading-language text ablation (thesis core) run first; the long ISIC and
# ConvNeXt-Large jobs run afterwards if the night is long enough.
# ---------------------------------------------------------------------------

[ -f requirements.txt ] || { echo "ERROR: must run from repo root"; exit 1; }

LOGDIR="results/overnight_logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"
: > "$SUMMARY"

log() { echo "$*" | tee -a "$SUMMARY"; }

log "==================================================================="
log "Overnight run STARTED: $(date '+%F %T')"
log "GPU: $(py -3.11 -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null)"
log "==================================================================="
log ""

run () {
  name="$1"; shift
  log ">>> [$name] START $(date '+%F %T')"
  log "    cmd: $*"
  start=$(date +%s)
  "$@" > "$LOGDIR/${name}.log" 2>&1
  code=$?
  dur=$(( $(date +%s) - start ))
  if [ $code -eq 0 ]; then st="OK"; else st="FAIL (exit $code)"; fi
  log "<<< [$name] $st  $((dur/60))m (${dur}s)  $(date '+%F %T')"
  log ""
}

# --- Core (thesis priority) ---
run ham_all           py -3.11 scripts/run_experiments.py --mode all
run ham_eng_metadata  py -3.11 scripts/run_experiments.py --mode vision_metadata --config configs/eng_metadata_ham10000.yaml
run ham_text_Orig     py -3.11 scripts/run_experiments.py --mode vision_text --config configs/text_ham10000.yaml
run ham_text_CFilt    py -3.11 scripts/run_experiments.py --mode vision_text --config configs/text_ham10000_CFilt.yaml
run ham_text_DFilt    py -3.11 scripts/run_experiments.py --mode vision_text --config configs/text_ham10000_DFilt.yaml
run ham_text_FFilt    py -3.11 scripts/run_experiments.py --mode vision_text --config configs/text_ham10000_FFilt.yaml

# --- Extended (run if time permits) ---
run isic_eng_all      py -3.11 scripts/run_experiments.py --mode all --config configs/eng_metadata_isic2020.yaml
run ham_large_all     py -3.11 scripts/run_experiments.py --mode all --config configs/large_ham10000.yaml
run isic_all          py -3.11 scripts/run_experiments.py --mode all --config configs/isic2020.yaml

# --- Aggregate everything for the morning ---
log "Regenerating comparison tables..."
py -3.11 -m src.evaluation.compare --results-dir results --out results/comparison.md > "$LOGDIR/_compare.log" 2>&1
log "Comparison -> results/comparison.md (+ .csv)"

log "==================================================================="
log "Overnight run FINISHED: $(date '+%F %T')"
log "==================================================================="
