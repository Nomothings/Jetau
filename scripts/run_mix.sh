#!/bin/bash
# Mixed six-task single-model training: Jet-0.6B on GPU0 (window 6),
# Jet-1.7B on GPU1 (window 2, proven 1.7B memory config), 600 steps each.
# Then evaluate each model on all six official test sets (GPU2).
cd /data/yangyuming/Long-Jev/code/memexp
L=/data/yangyuming/Long-Jev/logs/smoke
D=../../data/benchmarks/mix

nohup setsid bash -c "CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_jet_bs.py --task mix \
  --train $D/train.jsonl --dev $D/dev.jsonl \
  --checkpoint ../../checkpoints/NanoJev-unified --steps 600 --eval-every 50 --eval-episodes 60 \
  --note-window 6 --note-slots 16 --note-cap 4 --max-train-steps 20 --mem-fraction 0.9 \
  --out ../../models/jet06_mix >> $L/jet06_mix.log 2>&1; \
  RC=\$?; [ \$RC -eq 0 ] && touch $L/jet06_mix.done || touch $L/jet06_mix.FAILED" </dev/null >/dev/null 2>&1 &

nohup setsid bash -c "CUDA_VISIBLE_DEVICES=1 .venv/bin/python train_jet_bs.py --task mix \
  --train $D/train.jsonl --dev $D/dev.jsonl \
  --checkpoint ../../checkpoints/decision-qwen3-1.7b --steps 600 --eval-every 50 --eval-episodes 60 \
  --note-window 2 --note-slots 16 --note-cap 4 --max-train-steps 8 --mem-fraction 0.92 \
  --out ../../models/jet17_mix >> $L/jet17_mix.log 2>&1; \
  RC=\$?; [ \$RC -eq 0 ] && touch $L/jet17_mix.done || touch $L/jet17_mix.FAILED" </dev/null >/dev/null 2>&1 &

# evals once both finish
nohup setsid bash -c "while [ ! -f $L/jet06_mix.done ] || [ ! -f $L/jet17_mix.done ]; do sleep 120; done; \
  for SZ in 06 17; do for T in maze snake pokemon alfworld mind2web webshop; do \
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python eval_mem.py --arm c7 --task \$T \
      --checkpoint ../../models/jet\${SZ}_mix --test ../../data/benchmarks/\$T/test.jsonl \
      --limit-episodes 0 --out $L/testmem_\${T}_mix\${SZ}.json --mem-fraction 0.5 \
      >> $L/testmem_\${T}_mix\${SZ}.log 2>&1; done; done; touch $L/mix_evals.done" </dev/null >/dev/null 2>&1 &

echo mix-launched
