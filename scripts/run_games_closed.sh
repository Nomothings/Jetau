#!/bin/bash
# Closed-loop game grid: 3 tasks x {nomem, promptmem, jet06, jet17, mix06, mix17}.
cd /data/yangyuming/Long-Jev/code/memexp
L=/data/yangyuming/Long-Jev/logs/smoke
O=$L/closed
mkdir -p $O
CK=../../checkpoints/NanoJev-unified
EPmaze=60; EPsnake=50; EPpoke=100

run () {  # gpu task policy checkpoint modelname episodes
  local MN=${5:-base}
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python eval_games_closed.py --task $2 --policy $3 \
    --checkpoint $4 --model-name "$MN" --episodes $6 --out $O/${2}_${3}_${MN}.json \
    --mem-fraction 0.4 >> $O/${2}_${3}_${MN}.log 2>&1 \
    && echo "closed $2 $3 $MN ok" >> $O/progress.log
}

# GPU0: maze family
( run 0 maze nomem $CK "" $EPmaze; run 0 maze promptmem $CK "" $EPmaze
  run 0 maze jet ../../models/jet06_maze jet06_maze $EPmaze
  run 0 maze jet ../../models/jet17_maze jet17_maze $EPmaze
  run 0 maze jet ../../models/jet06_mix jet06_mix $EPmaze
  run 0 maze jet ../../models/jet17_mix jet17_mix $EPmaze ) &
# GPU1: snake family
( run 1 snake nomem $CK "" $EPsnake; run 1 snake promptmem $CK "" $EPsnake
  run 1 snake jet ../../models/jet06_snake jet06_snake $EPsnake
  run 1 snake jet ../../models/jet17_snake jet17_snake $EPsnake
  run 1 snake jet ../../models/jet06_mix jet06_mix $EPsnake
  run 1 snake jet ../../models/jet17_mix jet17_mix $EPsnake ) &
# GPU2: pokemon family
( run 2 pokemon nomem $CK "" $EPpoke; run 2 pokemon promptmem $CK "" $EPpoke
  run 2 pokemon jet ../../models/jet06_pokemon jet06_pokemon $EPpoke
  run 2 pokemon jet ../../models/jet17_pokemon jet17_pokemon $EPpoke
  run 2 pokemon jet ../../models/jet06_mix jet06_mix $EPpoke
  run 2 pokemon jet ../../models/jet17_mix jet17_mix $EPpoke ) &
wait
touch $O/all_games_closed.done
echo "games closed-loop done $(date)" >> $O/progress.log
