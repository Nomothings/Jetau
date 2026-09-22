#!/bin/bash
# Sync final artifacts from the original Long-Jev experiment tree into this
# repository. Intentionally left as a documented stub: the model bundles and
# result JSONs live on the experiment server, so this script is filled in by
# the author before the public release. Fill in and run from the repo root:
#
#   SRC=${SRC:-/data/yangyuming/Long-Jev}
#
#   # 1) 12 trained Jet bundles (Jet-0.6B and Jet-1.7B x 6 tasks):
#   for SZ in 06 17; do
#     for T in maze snake pokemon alfworld mind2web webshop; do
#       mkdir -p checkpoints/jet/jet${SZ}_${T}
#       cp -r "$SRC"/models/jet${SZ}_${T}/. checkpoints/jet/jet${SZ}_${T}/
#     done
#   done
#
#   # 2) Evaluation results (base_nomem_* / base_promptmem_* / testmem_* JSONs):
#   cp "$SRC"/logs/smoke/base_nomem_*.json    results/
#   cp "$SRC"/logs/smoke/base_promptmem_*.json results/
#   cp "$SRC"/logs/smoke/testmem_*.json       results/
#
#   # 3) Rebuild the final comparison table from the synced files:
#   python -m jet.collect_final
set -u
echo "sync_artifacts.sh is a stub - edit it to point at your Long-Jev tree, then uncomment the copies."
exit 0
