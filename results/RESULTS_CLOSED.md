## Official-protocol interactive results (0.6B)

Games: completion within the step budget (mean steps of successes). ALFWorld: interactive success rate, official unseen 134 games, 50-step limit. WebShop: interactive purchase, official reward. Baselines here are the training-free released checkpoint; the trained-baseline comparison lands with the in-flight runs.

| Task | No memory | Prompt memory | Jeτ-0.6B | Jeτ-0.6B-Mix |
|---|---:|---:|---:|---:|
| maze | 0% | - | 80% (15.1 st) | 55% (17.8 st) |
| snake | 0% | 0% | 88% (15.5 st) | 54% (15.6 st) |
| pokemon | 3% (21.0 st) | 1% (17.0 st) | 48% (15.5 st) | 45% (16.0 st) |
| alfworld | 0% | - | 76% (9.6 st) | 47% (12.2 st) |
| webshop | 10% | 12% | 43% | 42% |

Mind2Web (official metric is step-level): no memory 0.132 / prompt 0.133 / Jeτ-0.6B 0.631 / Mix 0.546.

WebShop strict success (r=1.0): no memory 10.2% / prompt 12.0% / Jeτ-0.6B 43.2%.
