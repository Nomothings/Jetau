#!/bin/bash
# Arm test evals for all 12 fleet models (Jet-0.6B/1.7B x 6 tasks) with timing.
# Runs sequentially per task as each training finishes; light (inference only).
# Env overrides: PYTHON, LOGS (default logs), EVAL_GPU (default 1),
#   OUT (results dir, default results).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python}
L=${LOGS:-logs}
OUT=${OUT:-results}
mkdir -p "$OUT"

wait_and_eval () {  # done_marker size task
  while [ ! -f $L/$1 ]; do sleep 90; done
  CUDA_VISIBLE_DEVICES=${EVAL_GPU:-1} $PYTHON -m jet.eval_mem --arm c7 --task $3 \
    --checkpoint checkpoints/jet/jet${2}_$3 --test data/$3/test.jsonl \
    --limit-episodes 0 --out $OUT/testmem_${3}_jet${2}.json --mem-fraction 0.5 \
    >> $L/testmem_${3}_jet${2}.log 2>&1 && echo "eval jet$2 $3 done" >> $L/fleet_eval.log
}

for T in maze snake pokemon alfworld mind2web webshop; do wait_and_eval jet06_${T}.done 06 $T & done
for T in maze snake pokemon alfworld mind2web webshop; do wait_and_eval jet17_${T}.done 17 $T & done
wait
echo "fleet evals all done $(date)" >> $L/fleet_eval.log
