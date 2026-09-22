#!/bin/bash
# Jet fleet dispatcher: 12 runs (Jet-0.6B / Jet-1.7B x 6 tasks, 300 steps,
# batched-streaming trainer). Launches the next queued job on any GPU whose
# memory is free, exits when all done markers exist.
# Env overrides: PYTHON, LOGS (default logs), GPUS (default "0 1 2").
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python}
L=${LOGS:-logs}
GPUS=${GPUS:-"0 1 2"}
mkdir -p "$L" checkpoints/jet

declare -A CKPT=( [06]=checkpoints/NanoJev-unified [17]=checkpoints/decision-qwen3-1.7b )
declare -A FRAC=( [06]=0.9 [17]=0.9 )
JOBS="06_maze 06_snake 06_pokemon 06_alfworld 06_mind2web 06_webshop 17_maze 17_snake 17_pokemon 17_alfworld 17_mind2web 17_webshop"

free_gpu () {
  for G in $GPUS; do
    local USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)
    [ "$USED" -lt 2000 ] && { echo $G; return; }
  done
  echo ""
}

for JOB in $JOBS; do
  SZ=${JOB%%_*}; T=${JOB#*_}
  while [ ! -f $L/jet${SZ}_${T}.done ]; do
    G=$(free_gpu)
    if [ -n "$G" ]; then
      # claim via a running marker to avoid double-launch races
      ( set -o noclobber; > $L/jet${SZ}_${T}.running ) 2>/dev/null || { sleep 45; continue; }
      echo "$(date) jet$SZ $T -> GPU $G" >> $L/fleet.log
      setsid nohup bash -c "CUDA_VISIBLE_DEVICES=$G $PYTHON -m jet.train_jet_bs --task $T \
        --train data/$T/train.jsonl --dev data/$T/dev.jsonl \
        --checkpoint ${CKPT[$SZ]} --steps 300 --eval-every 25 --eval-episodes 24 \
        --note-window 6 --note-slots 16 --note-cap 4 --max-train-steps 20 \
        --mem-fraction ${FRAC[$SZ]} --out checkpoints/jet/jet${SZ}_${T} \
        >> $L/jet${SZ}_${T}.log 2>&1; touch $L/jet${SZ}_${T}.done; rm -f $L/jet${SZ}_${T}.running" \
        </dev/null >/dev/null 2>&1 &
      sleep 90
    else
      sleep 45
    fi
  done
done
echo "$(date) jet fleet all done" >> $L/fleet.log
