# Evaluation results

## Step-level decisions

Step accuracy is the number of target actions selected divided by all scored
decision points across the test episodes. The first two columns use the
released NanoJev checkpoint without task training. Jeτ-0.6B is trained on
each task's episode data.

| Task | Observation only | Text history | Jeτ-0.6B |
|---|---:|---:|---:|
| Maze | 39.2% | 37.4% | **83.2%** |
| Snake | 55.3% | 35.1% | **91.0%** |
| Pokémon | 31.0% | 26.0% | **79.6%** |
| ALFWorld | 18.0% | 40.2% | **86.9%** |
| Mind2Web | 13.2% | 13.3% | **63.1%** |
| WebShop | 45.1% | 46.8% | **82.3%** |

## Complete tasks

The observation-only model and Jeτ in this table both receive task training.
Each result comes from interactive evaluation through the corresponding
environment. Mind2Web uses annotated element selection as its benchmark
metric and appears in the step-level table above.

| Task | Metric | Observation only | Jeτ-0.6B |
|---|---|---:|---:|
| Maze | Completion rate | 11.7% | **80.0%** |
| Snake | Completion rate | **92.0%** | 88.0% |
| Pokémon | Win rate | **53.0%** | 48.0% |
| ALFWorld | Completion rate | 45.5% | **76.1%** |
| WebShop | Mean reward | **0.680** | 0.675 |

The maze runs use a 64-step budget; successful Jeτ runs average 15.1 steps.
ALFWorld uses 134 unseen household games and a 50-step budget. WebShop uses
the offline shopping environment described in [the data notes](../data/README.md).

To produce new result files, use `jet/eval_mem.py` for step-level Jeτ accuracy,
`jet/eval_acc.py` and `jet/eval_promptmem.py` for the two baselines, and the
`jet/eval_*_closed.py` entry points for interactive evaluation. The individual
development logs, ablations, and raw evaluation traces are omitted from this
release.
