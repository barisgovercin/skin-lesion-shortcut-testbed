#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full shortcut-learning sweep (5 seeds): HAM10000 then ISIC 2020.
# Each: dose-response (vary leakage_p) + mitigation bake-off, then plots.
# Sequential, logged, CONTINUES past failures. Safe to leave unattended.
# ---------------------------------------------------------------------------
[ -f requirements.txt ] || { echo "ERROR: run from repo root"; exit 1; }

LOGDIR="results/overnight_logs"; mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/shortcut_summary.txt"; : > "$SUMMARY"
log(){ echo "$*" | tee -a "$SUMMARY"; }

log "==================================================================="
log "Shortcut sweep STARTED: $(date '+%F %T')"
log "GPU: $(py -3.11 -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null)"
log "==================================================================="
log ""

run(){ name="$1"; shift
  log ">>> [$name] START $(date '+%F %T')"
  s=$(date +%s); "$@" > "$LOGDIR/${name}.log" 2>&1; c=$?; d=$(( $(date +%s)-s ))
  [ $c -eq 0 ] && st="OK" || st="FAIL (exit $c)"
  log "<<< [$name] $st  $((d/60))m  $(date '+%F %T')"; log ""
}

PLOT='from pathlib import Path
from src.evaluation.plots import plot_dose_response, plot_bakeoff
ds = __import__("sys").argv[1]
d=f"results/shortcut_{ds}_dose.json"; b=f"results/shortcut_{ds}_bakeoff.json"
if Path(d).exists(): plot_dose_response(d, f"results/shortcut_{ds}_dose.png")
if Path(b).exists(): plot_bakeoff(b, f"results/shortcut_{ds}_bakeoff.png")
print("plots done for", ds)'

# --- HAM10000 (core) ---
run shortcut_ham_dose     py -3.11 scripts/run_shortcut.py --config configs/shortcut_ham10000.yaml --mode dose
run shortcut_ham_bakeoff  py -3.11 scripts/run_shortcut.py --config configs/shortcut_ham10000.yaml --mode bakeoff
run plots_ham             py -3.11 -c "$PLOT" ham10000

# --- ISIC 2020 (generalisation) ---
run shortcut_isic_dose    py -3.11 scripts/run_shortcut.py --config configs/shortcut_isic2020.yaml --mode dose
run shortcut_isic_bakeoff py -3.11 scripts/run_shortcut.py --config configs/shortcut_isic2020.yaml --mode bakeoff
run plots_isic            py -3.11 -c "$PLOT" isic2020

log "==================================================================="
log "Shortcut sweep FINISHED: $(date '+%F %T')"
log "==================================================================="
