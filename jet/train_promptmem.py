"""Trained prompt-text-memory baseline (0.6B): identical recipe to the
no-memory trainer (train_generic) except the state carries a sliding-window
text history of the episode (same rendering as eval_promptmem). Supervised
per-step CE on the teacher action; no latent memory mechanisms.

Config parity with Jet specialists: same NanoJev-unified warm start, same
task train set, 300 gradient updates, seed 17, AdamW (1e-5/1e-4).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jet.bench_common import build_generic_example, load_episodes
from jet.common import load_decision_model, save_bundle


def build_state(hist, obs):
    state = ""
    if hist:
        state = "Recent steps (oldest first):\n" + "\n".join(hist) + "\n"
    return state + f"Current observation:\n{obs}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--checkpoint", default="../../checkpoints/NanoJev-unified")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--train-episodes", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--batch-questions", type=int, default=16)
    p.add_argument("--max-hist-chars", type=int, default=800)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--mem-fraction", type=float, default=0.6)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    runtime = load_decision_model(a.checkpoint, mem_fraction=a.mem_fraction)
    model, tokenizer = runtime.model, runtime.tokenizer
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device

    train_eps = load_episodes(a.train, a.train_episodes)
    dev_eps = load_episodes(a.dev)[: a.eval_episodes]
    print(f"promptmem task={a.task} train={len(train_eps)} dev={len(dev_eps)}", flush=True)

    def episode_examples(ep):
        """Per-step examples with sliding-window text history (matches
        eval_promptmem's rendering and truncation)."""
        hist, out = [], []
        for s in ep["steps"]:
            state = build_state(hist, s["obs"])
            while True:
                try:
                    ex = build_generic_example(tokenizer, a.task, state,
                                               s["candidates"], runtime.limit)
                    break
                except ValueError:
                    if hist:
                        hist.pop(0)
                    else:
                        s_obs = s["obs"]
                        state = f"Current observation:\n{s_obs[:max(600, len(s_obs)-400)]}"
                        try:
                            ex = build_generic_example(tokenizer, a.task, state,
                                                       s["candidates"], runtime.limit)
                        except ValueError:
                            ex = build_generic_example(
                                tokenizer, a.task,
                                f"Current observation:\n{s_obs[:600]}",
                                s["candidates"], runtime.limit)
                        break
            ex["target"] = ex["candidate_ids"].index(s["action"])
            out.append(ex)
            hist.append(f"Obs: {s['obs'][:a.max_hist_chars]}\nDid: {s['action']} -> {s['event']}")
        return out

    @torch.no_grad()
    def dev_eval():
        model.eval()
        total, n, acc = 0.0, 0, 0
        for ep in dev_eps:
            for ex in episode_examples(ep):
                logits, _ = model([ex], pad_id)
                k = len(ex["candidate_ids"])
                total += float(torch.nn.functional.cross_entropy(
                    logits[0][:k].float().unsqueeze(0),
                    torch.tensor([ex["target"]], device=device)))
                acc += int(logits[0][:k].argmax().item() == ex["target"])
                n += 1
        model.train()
        return total / max(n, 1), acc / max(n, 1)

    body = list(model.backbone.parameters())
    head = [q for name, q in model.named_parameters() if not name.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.backbone_lr},
                             {"params": head, "lr": a.head_lr}], weight_decay=0.01)
    ce0, acc0 = dev_eval()
    print(f"initial dev CE {ce0:.4f} acc {acc0:.3f}", flush=True)
    save_bundle(model, tokenizer, a.checkpoint, a.out)
    best, best_step, logs = ce0, 0, []
    order = list(range(len(train_eps)))
    random.Random(a.seed).shuffle(order)
    started, pointer, step_losses = time.time(), 0, []
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        batch, seen = [], 0
        while seen < a.batch_questions:
            exs = episode_examples(train_eps[order[pointer % len(order)]])
            pointer += 1
            batch.extend(exs)
            seen += len(exs)
        batch = batch[: a.batch_questions + 8]
        n_total = len(batch)
        for mi in range(0, n_total, 4):          # micro-batches, grad accumulation
            chunk = batch[mi:mi + 4]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(chunk, pad_id)
                losses = [torch.nn.functional.cross_entropy(
                    row[:len(ex["candidate_ids"])].float().unsqueeze(0),
                    torch.tensor([ex["target"]], device=device))
                    for row, ex in zip(logits, chunk)]
                loss = torch.stack(losses).mean() * (len(chunk) / n_total)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step_losses.append(float(loss) / (len(chunk) / n_total))
        if step % 10 == 0:
            print(f"step {step} loss {sum(step_losses)/len(step_losses):.4f} "
                  f"elapsed {time.time()-started:.0f}s", flush=True)
            step_losses = []
        if step % a.eval_every == 0 or step == a.steps:
            ce, acc = dev_eval()
            logs.append({"step": step, "dev_ce": ce, "dev_acc": acc})
            print(f"step {step} dev CE {ce:.4f} acc {acc:.3f}", flush=True)
            if ce < best:
                best, best_step = ce, step
                save_bundle(model, tokenizer, a.checkpoint, a.out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "train_log.json").write_text(json.dumps(
        {"trainer": "promptmem", "task": a.task, "config": vars(a),
         "initial_dev_ce": ce0, "initial_dev_acc": acc0,
         "best_step": best_step, "best_dev_ce": best, "logs": logs}, indent=2))
    print(json.dumps({"done": a.out, "best_step": best_step, "best_dev_ce": best}))


if __name__ == "__main__":
    main()
