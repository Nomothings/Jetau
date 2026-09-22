"""Step-level accuracy evaluation on unified-format episodes (teacher-forced C1).

Works zero-shot with the released NanoJev checkpoint or with any bundle saved
by train_generic.py. Reports top-1 accuracy over each step's candidate set.
"""
import argparse
import json
from pathlib import Path

import torch

from jet.bench_common import episodes_to_examples, load_episodes
from jet.common import load_decision_model


@torch.no_grad()
def accuracy(model, examples, pad_id, device, micro_size=8, token_budget=16384):
    model.eval()
    n, acc = 0, 0
    for i in range(0, len(examples), micro_size):
        batch = examples[i:i + micro_size]
        widest = max(max(map(len, e["leaf_tokens"])) for e in batch)
        if widest * 4 * len(batch) > token_budget:
            batch = batch[: max(1, token_budget // (widest * 4))]
        logits, _ = model(batch, pad_id)
        for row, ex in zip(logits, batch):
            k = len(ex["candidate_ids"])
            acc += int(row[:k].argmax().item() == ex["target"])
            n += 1
    return acc / max(n, 1), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--checkpoint", default="checkpoints/NanoJev-unified")
    p.add_argument("--limit-steps", type=int, default=0, help="cap eval steps (0=all)")
    p.add_argument("--test-episodes", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--mem-fraction", type=float, default=0.65)
    a = p.parse_args()

    runtime = load_decision_model(a.checkpoint, mem_fraction=a.mem_fraction)
    model, tokenizer = runtime.model, runtime.tokenizer
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device

    eps = load_episodes(a.test, a.test_episodes)
    examples, skipped = episodes_to_examples(eps, tokenizer, a.task, runtime.limit,
                                             max_examples=a.limit_steps)
    mean_k = sum(len(e["candidate_ids"]) for e in examples) / max(len(examples), 1)
    import time
    t0 = time.perf_counter()
    acc, n = accuracy(model, examples, pad_id, device)
    wall = time.perf_counter() - t0
    result = {"task": a.task, "checkpoint": a.checkpoint, "test": a.test,
              "n_episodes": len(eps), "n_steps_evaluated": n, "skipped_overlong": skipped,
              "step_acc": acc, "random_baseline": 1.0 / mean_k, "mean_candidates": mean_k,
              "ms_per_step": wall / max(n, 1) * 1000, "steps_per_sec": n / max(wall, 1e-9)}
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
