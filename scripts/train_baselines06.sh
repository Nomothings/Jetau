#!/bin/bash
# Train the two baseline arms (no-memory + promptmem) for all six tasks,
# 0.6B, config parity with Jet specialists (NanoJev-unified init, same train
# sets, 300 updates, seed 17). Free-GPU dispatcher with success markers.
cd /data/yangyuming/Long-Jev/code/memexp
L=/data/yangyuming/Long-Jev/logs/smoke
CK=../../checkpoints/NanoJev-unified

JOBS="nomem_maze nomem_snake nomem_pokemon nomem_alfworld nomem_mind2web nomem_webshop pmem_maze pmem_snake pmem_pokemon pmem_alfworld pmem_mind2web pmem_webshop"

free_gpu () {
  for G in 0 1 2; do
    local USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)
     [ "$USED" -lt 30000 ] && { echo $G; return; }
  done
  echo ""
}

while :; do
  ALLDONE=1
  for JOB in $JOBS; do
    ARM=${JOB%%_*}; T=${JOB#*_}
    [ -f $L/${JOB}.done ] && continue
    ALLDONE=0
    [ -f $L/${JOB}.running ] && continue
    [ -f $L/${JOB}.FAILED ] && continue
    G=$(free_gpu)
    if [ -n "$G" ]; then
      ( set -o noclobber; > $L/${JOB}.running ) 2>/dev/null || continue
      echo "$(date) $JOB -> GPU $G" >> $L/baseline_train.log
      setsid nohup bash -c "CUDA_VISIBLE_DEVICES=$G .venv/bin/python \
        $( [ $ARM = nomem ] && echo train_generic.py || echo train_promptmem.py ) \
        --task $T --train ../../data/benchmarks/$T/train.jsonl \
        --dev ../../data/benchmarks/$T/dev.jsonl \
        --checkpoint $CK --steps 300 --eval-every 50 --seed 17 --mem-fraction 0.38 \
        $( [ $ARM = nomem ] && echo '--batch-questions 16' || echo '--batch-questions 16 --eval-episodes 24' ) \
        --out ../../models/${ARM}06_$T >> $L/${JOB}.log 2>&1; \
        RC=\$?; [ \$RC -eq 0 ] && touch $L/${JOB}.done || touch $L/${JOB}.FAILED; \
        rm -f $L/${JOB}.running" </dev/null >/dev/null 2>&1 &
      sleep 75
    fi
  done
  [ "$ALLDONE" = "1" ] && break
  sleep 45
done
echo "$(date) baseline training fleet done" >> $L/baseline_train.log
