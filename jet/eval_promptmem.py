#!/usr/bin/env python3
"""Prompt-text memory baseline (training-free): episode history is rendered as
plain text inside the prompt; when the token budget overflows, a sliding
window drops the oldest history steps (whole entries). Uses the released
NanoJev checkpoint unchanged -- this is the "pure prompt memory" comparison
strategy. Reports step accuracy, mean CE, and inference speed.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from jet.bench_common import build_generic_example, load_episodes
from jet.common import load_decision_model


def render_hist(entries):
    if not entries:
        return ""
    return "Recent steps (oldest first):\n" + "\n".join(entries) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--checkpoint", default="checkpoints/NanoJev-unified")
    p.add_argument("--out", default="")
    p.add_argument("--max-hist-chars", type=int, default=800,
                   help="per-entry observation truncation inside history")
    p.add_argument("--mem-fraction", type=float, default=0.35)
    a = p.parse_args()

    runtime = load_decision_model(a.checkpoint, mem_fraction=a.mem_fraction)
    model, tokenizer = runtime.model, runtime.tokenizer
    model.eval()
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    budget = runtime.limit

    eps = load_episodes(a.test, 0)
    correct, total_ce, n = 0, 0.0, 0
    window_sizes = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for ep in eps:
            hist = []
            for s in ep["steps"]:
                state = render_hist(hist) + f"Current observation:\n{s['obs']}"
                ex = None
                while ex is None:
                    try:
                        ex = build_generic_example(tokenizer, a.task, state,
                                                   s["candidates"], budget)
                    except ValueError:
                        if hist:
                            hist.pop(0)  # sliding window: drop oldest entry
                        else:
                            # obs itself too long: hard-truncate and retry
                            s_obs = s["obs"]
                            cut = max(600, len(s_obs) - 400)
                            state = f"Current observation:\n{s_obs[:cut]}"
                            try:
                                ex = build_generic_example(tokenizer, a.task, state,
                                                           s["candidates"], budget)
                            except ValueError:
                                ex = build_generic_example(
                                    tokenizer, a.task, f"Current observation:\n{s_obs[:600]}",
                                    s["candidates"], budget)
                logits, _ = model([ex], pad_id)
                row = logits[0]
                k = len(ex["candidate_ids"])
                target = ex["candidate_ids"].index(s["action"])
                total_ce += float(torch.nn.functional.cross_entropy(
                    row[:k].float().unsqueeze(0), torch.tensor([target], device=device)))
                correct += int(row[:k].argmax().item() == target)
                n += 1
                window_sizes.append(len(hist))
                obs_line = f"Obs: {s['obs'][:a.max_hist_chars]}"
                hist.append(f"{obs_line}\nDid: {s['action']} -> {s['event']}")
    wall = time.perf_counter() - t0
    result = {"strategy": "promptmem", "task": a.task, "checkpoint": a.checkpoint,
              "test": a.test, "n_episodes": len(eps), "n_steps": n,
              "step_acc": correct / max(n, 1), "mean_ce": total_ce / max(n, 1),
              "ms_per_step": wall / max(n, 1) * 1000, "steps_per_sec": n / max(wall, 1e-9),
              "mean_history_window": sum(window_sizes) / max(len(window_sizes), 1),
              "max_length_budget": budget}
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
