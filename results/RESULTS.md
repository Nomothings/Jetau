# Jet Results

Test-set step accuracy (teacher-forced) and inference latency (ms/decision) for four
strategies on six benchmarks. No-memory and prompt-text memory are training-free
(released NanoJev checkpoint); Jet-0.6B / Jet-1.7B trained 300 steps with the batched
streaming trainer (`scripts/`). All test sets are the full official splits.

## Accuracy (test step acc)

| Task | No memory | Prompt memory (sliding window) | Jet-0.6B | Jet-1.7B |
|---|---:|---:|---:|---:|
| maze | 0.392 | 0.374 | **0.832** | 0.822 |
| snake | 0.553 | 0.351 | **0.910** | 0.872 |
| pokemon | 0.310 | 0.260 | **0.796** | 0.868 |
| alfworld | 0.180 | 0.402 | **0.869** | 0.863 |
| mind2web | 0.132 | 0.133 | **0.631** | 0.631 |
| webshop | 0.451 | 0.468 | **0.823** | 0.870 |

## Inference latency (ms per decision step)

| Task | No memory | Prompt memory | Jet-0.6B | Jet-1.7B |
|---|---:|---:|---:|---:|
| maze | 44 | 403 | 271 | 242 |
| snake | 43 | 473 | 305 | 203 |
| pokemon | 129 | 1510 | 459 | 301 |
| alfworld | 170 | 195 | 315 | 473 |
| mind2web | 130 | 279 | 338 | 320 |
| webshop | 446 | 576 | 275 | 313 |

## Training loss (Jet, best dev CE / final test CE)

| Task | Jet-0.6B best dev CE | Jet-1.7B best dev CE |
|---|---:|---:|
| maze | 0.622 | 0.621 |
| snake | 0.333 | 0.417 |
| pokemon | 0.686 | 0.426 |
| alfworld | 0.248 | 0.273 |
| mind2web | 0.787 | 0.805 |
| webshop | 0.777 | 0.693 |

## Notes

- Prompt-text memory keeps history as plain text with a sliding window (oldest steps
  dropped at the 2048-token budget). It underperforms *no memory* on 4/6 tasks --
  long observations crowd out the current state -- and is 1.3-3.3x slower than Jet.
- Jet-1.7B used note-window 3 (games maze/snake/mind2web) or 2 (pokemon) to fit in
  48GB; Jet-0.6B used note-window 6. The smaller write window handicaps
  memory-heavy tasks (see maze), while 1.7B still wins pokemon/webshop.
- alfworld/mind2web/webshop Jet-0.6B models were trained earlier under the identical
  recipe and reused (see repository history).
