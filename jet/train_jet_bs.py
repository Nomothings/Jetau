#!/usr/bin/env python3
"""Train Jeτ with recurrent latent state and batched candidate scoring."""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from jet.bench_common import POLICY_QUESTION, TASK_HEADERS, load_episodes
from jet.common import leaf_text, load_decision_model, save_bundle, tokenize_segments
from jet.latent_state import (LatentStateCache, LatentStateWriter,
                              detach_cache, step_outcome, write_latent_state)
from transformers import DynamicCache


class BroadcastNoStoreCache(DynamicCache):
    """Wraps a batch-1 cache for batched leaf scoring: broadcasts the shared KV
    prefix to the batch size and concatenates the new tokens functionally,
    never writing back (the persistent cache stays untouched)."""

    def __init__(self, base_cache, batch):
        super().__init__()
        self._base = base_cache
        self._batch = batch
        for i, layer in enumerate(base_cache.layers):
            super().update(layer.keys, layer.values, i)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        base = self._base.layers[layer_idx]
        k = torch.cat([base.keys.expand(self._batch, *([-1] * (base.keys.dim() - 1))),
                       key_states], dim=-2)
        v = torch.cat([base.values.expand(self._batch, *([-1] * (base.values.dim() - 1))),
                       value_states], dim=-2)
        return k, v


def score_leaves_bs(model, leaf_ids, cache, device, grad=True, pos_start=None):
    """Batched leaf scoring: one padded forward for all K candidate leaves.

    pos_start: absolute position of the first leaf token (required for caches
    with position gaps)."""
    past_len = cache.get_seq_length()
    K = len(leaf_ids)
    maxlen = max(len(x) for x in leaf_ids)
    pad = leaf_ids[0][-1]
    tokens = torch.full((K, maxlen), pad, dtype=torch.long, device=device)
    attn = torch.ones((K, past_len + maxlen), dtype=torch.long, device=device)
    for i, ids in enumerate(leaf_ids):
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, past_len + len(ids):] = 0
    pos = torch.arange(pos_start, pos_start + maxlen, device=device).unsqueeze(0).expand(K, maxlen)
    view = BroadcastNoStoreCache(cache, K)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model.backbone(input_ids=tokens, attention_mask=attn, position_ids=pos,
                             past_key_values=view, use_cache=False)
    last = torch.stack([out.last_hidden_state[i, len(ids) - 1]
                        for i, ids in enumerate(leaf_ids)]).unsqueeze(0)
    h = model.norm(last)
    z = model.scalar(h).squeeze(-1).float()
    if getattr(model, "set_head", None) == "attention":
        valid = torch.ones((1, K), dtype=torch.bool, device=device)
        log_k = valid.sum(-1).float().log()[:, None, None].expand(-1, K, 1)
        u = model.set_project(torch.cat([h, log_k.to(h.dtype)], dim=-1))
        mixed, _ = model.set_attention(u, u, u, key_padding_mask=~valid, need_weights=False)
        z = z + model.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
    return z[0]


def episode_losses_jet_bs(model, tokenizer, task, ep, device, max_length, writer,
                          train=True, scale=1.0, note_slots=16, note_window=12,
                          note_cap=4, max_train_steps=None):
    header = [TASK_HEADERS[task], f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]
    n = len(ep["steps"])
    ec = LatentStateCache(model, tokenizer, device)
    ec.add_segment("header", tokenize_segments(tokenizer, header), grad=train)
    losses, actions_ok, window_losses, steps_in_window = [], 0, [], 0
    limit = n if (max_train_steps is None or not train) else min(n, max_train_steps)
    for i, s in enumerate(ep["steps"][:limit]):
        seg = tokenize_segments(tokenizer, [f"State:\n{s['obs']}\n"])
        if ec.length() + len(seg) + 64 > max_length:
            break
        if train and steps_in_window >= note_window:
            if window_losses:
                (sum(window_losses) / max(n, 1) * scale).backward()
            window_losses, steps_in_window = [], 0
            ec.cache = detach_cache(ec.cache)
        ec.add_segment("step", seg, grad=train)
        keys = list(s["candidates"])
        leaf_ids = [tokenizer.encode(leaf_text(k, s["candidates"][k]), add_special_tokens=False)
                    + [tokenizer.eos_token_id] for k in keys]
        logits = score_leaves_bs(model, leaf_ids, ec.cache, device, grad=train, pos_start=ec.pos)
        target = torch.tensor([keys.index(s["action"])], device=device)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0).float(), target)
        if train:
            window_losses.append(loss)
        else:
            actions_ok += int(logits.argmax().item() == target.item())
        losses.append(float(loss.detach()))
        ec.add_segment("step", tokenize_segments(
            tokenizer, [step_outcome(s["action"], s["event"])]), grad=train)
        steps_in_window += 1
        if steps_in_window >= note_window:
            write_latent_state(model, writer, ec, device, grad=train)
            ec.rebuild(note_cap)
    if train and window_losses:
        (sum(window_losses) / max(n, 1) * scale).backward()
    return losses, actions_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--checkpoint", default="checkpoints/NanoJev-unified")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--train-episodes", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--note-slots", type=int, default=16)
    p.add_argument("--note-window", type=int, default=6)
    p.add_argument("--note-cap", type=int, default=4)
    p.add_argument("--max-train-steps", type=int, default=20)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--mem-fraction", type=float, default=0.9)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    runtime = load_decision_model(a.checkpoint, mem_fraction=a.mem_fraction)
    model, tokenizer = runtime.model, runtime.tokenizer
    device = next(model.parameters()).device
    max_length = runtime.limit
    note_token_id = tokenizer.encode("|", add_special_tokens=False)[0]
    writer = LatentStateWriter(model.backbone.config, a.note_slots).to(device)
    with torch.no_grad():
        base = model.backbone.get_input_embeddings().weight[note_token_id]
        writer.note_embed.copy_(base.unsqueeze(0) + 0.01 * torch.randn_like(writer.note_embed))
    model.writer = writer

    train_eps = load_episodes(a.train, a.train_episodes)
    dev_eps = load_episodes(a.dev)[: a.eval_episodes]
    print(f"jet-bs task={a.task} train={len(train_eps)} dev={len(dev_eps)}", flush=True)

    def dev_eval():
        model.eval()
        total, n, ok = 0.0, 0, 0
        with torch.no_grad():
            for ep in dev_eps:
                losses, good = episode_losses_jet_bs(
                    model, tokenizer, a.task, ep, device, max_length, writer,
                    train=False, note_slots=a.note_slots, note_window=a.note_window,
                    note_cap=a.note_cap)
                total += sum(losses)
                n += len(losses)
                ok += good
        model.train()
        return total / max(n, 1), ok / max(n, 1)

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
    started = time.time()
    step_losses = []
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        ep = train_eps[order[step % len(order)]]
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses, _ = episode_losses_jet_bs(
                    model, tokenizer, a.task, ep, device, max_length, writer,
                    train=True, scale=1.0, note_slots=a.note_slots,
                    note_window=a.note_window, note_cap=a.note_cap,
                    max_train_steps=a.max_train_steps)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"step {step}: OOM, skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        torch.cuda.empty_cache()
        if losses:
            step_losses.append(sum(losses) / len(losses))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 and step_losses:
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
        {"trainer": "jet-bs", "task": a.task, "config": vars(a),
         "initial_dev_ce": ce0, "initial_dev_acc": acc0,
         "best_step": best_step, "best_dev_ce": best, "logs": logs}, indent=2))
    print(json.dumps({"done": a.out, "best_step": best_step, "best_dev_ce": best}))


if __name__ == "__main__":
    main()
