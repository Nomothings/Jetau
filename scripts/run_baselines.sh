#!/bin/bash
# Phase-A baselines (training-free), full official test sets, with timing:
#   no-memory (jet.eval_acc) on 6 tasks and prompt-text sliding-window memory
#   (jet.eval_promptmem) on 6 tasks, launched in parallel.
# Env overrides: PYTHON, NOMEM_GPU (default 0), PROMPTMEM_GPU (default 2),
#   CKPT (default checkpoints/NanoJev-unified), L (output dir, default results).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python}
L=${L:-results}
CKPT=${CKPT:-checkpoints/NanoJev-unified}
mkdir -p "$L"

nohup setsid bash -c 'for T in maze snake pokemon alfworld mind2web webshop; do
  CUDA_VISIBLE_DEVICES='"${NOMEM_GPU:-0}"' '"$PYTHON"' -m jet.eval_acc --task $T \
    --test data/$T/test.jsonl --checkpoint '"$CKPT"' \
    --out '"$L"'/base_nomem_$T.json --mem-fraction 0.35 \
    >> '"$L"'/base_nomem.log 2>&1; done; touch '"$L"'/base_nomem.done' </dev/null >/dev/null 2>&1 &

nohup setsid bash -c 'for T in maze snake pokemon alfworld mind2web webshop; do
  CUDA_VISIBLE_DEVICES='"${PROMPTMEM_GPU:-2}"' '"$PYTHON"' -m jet.eval_promptmem --task $T \
    --test data/$T/test.jsonl --checkpoint '"$CKPT"' \
    --out '"$L"'/base_promptmem_$T.json --mem-fraction 0.35 \
    >> '"$L"'/base_promptmem.log 2>&1; done; touch '"$L"'/base_promptmem.done' </dev/null >/dev/null 2>&1 &

echo baselines-launched
