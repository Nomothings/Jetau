"""C3-C7 memory-arm trainer for any benchmark in the unified episode JSONL format.

This is the ablation bench. Arms:
  c3 -- linear transcript KV with windowed BPTT (train_c3.py mechanics)
  c4 -- two-tier bounded memory (train_c4.py mechanics)
  c5 -- unified compression: same as c4 but keep_steps=0; after each window the
        persistent memory is header + note slots only (no raw recent steps).
  c6 -- latent write head: note slots are free continuous embeddings forwarded
        over the cache (in-distribution base KV), plus a gated low-rank delta
        projected directly from the last hidden state (bypasses the tokenizer).
        Gates start at zero, so training begins exactly at c4-style behavior
        and can learn to write pure latent content.
  c7 -- Jet, the final method: c5's fully-unified structure (single latent
        tier, keep_steps=0) with c6's latent write head. Canonical entry
        point: train_jet.py (this arm stays here for ablation completeness).

Adapted from the grid-validated trainers; unified-format specifics:
task header from bench_common.TASK_HEADERS, candidates in dict order,
step outcome text from the recorded `event` field.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from jet.bench_common import TASK_HEADERS, load_episodes
from jet.common import leaf_text, load_decision_model, save_bundle, tokenize_segments
from jet.stream_core import (NOTE_PROMPT, EpisodeCache, detach_cache, forward_segment,
                             score_leaves)
from jet.vendor.unified_game_pipeline import POLICY_QUESTION
from transformers import DynamicCache


def step_outcome(action, event):
    return f"Decision taken: {action}. Result: {event}\n"


def episode_losses_c3(model, tokenizer, task, ep, device, max_length, train=True, scale=1.0,
                      bptt_window=8, max_train_steps=40):
    header = [TASK_HEADERS[task], f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]
    n = min(len(ep["steps"]), max_train_steps if train else len(ep["steps"]))
    cache = forward_segment(model, tokenize_segments(tokenizer, header), None, device,
                            grad=train).past_key_values
    losses, actions_ok, window_losses, window_start = [], 0, [], 0
    for i, s in enumerate(ep["steps"]):
        if i >= n:
            break
        seg = tokenize_segments(tokenizer, [f"State:\n{s['obs']}\n"])
        if cache.get_seq_length() + len(seg) + 64 > max_length:
            break
        if train and i > 0 and i - window_start >= bptt_window:
            if window_losses:
                (sum(window_losses) / max(n, 1) * scale).backward()
            window_losses, window_start = [], i
            cache = detach_cache(cache)
        cache = forward_segment(model, seg, cache, device, grad=train).past_key_values
        keys = list(s["candidates"])
        leaf_ids = [tokenizer.encode(leaf_text(k, s["candidates"][k]), add_special_tokens=False)
                    + [tokenizer.eos_token_id] for k in keys]
        logits = score_leaves(model, leaf_ids, cache, device, grad=train)
        target = torch.tensor([keys.index(s["action"])], device=device)
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0).float(), target)
        if train:
            window_losses.append(loss)
        else:
            actions_ok += int(logits.argmax().item() == target.item())
        losses.append(float(loss.detach()))
        cache = forward_segment(model, tokenize_segments(
            tokenizer, [step_outcome(s["action"], s["event"])]), cache, device, grad=train).past_key_values
    if train and window_losses:
        (sum(window_losses) / max(n, 1) * scale).backward()
    return losses, actions_ok


def episode_losses_c4(model, tokenizer, task, ep, device, max_length, train=True, scale=1.0,
                      note_slots=16, note_window=12, note_cap=4, keep_steps=2,
                      note_prompt_ids=None, note_token_id=None, max_train_steps=None):
    header = [TASK_HEADERS[task], f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]
    n = len(ep["steps"])
    ec = EpisodeCache(model, tokenizer, device, None)
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
        logits = score_leaves(model, leaf_ids, ec.cache, device, grad=train, pos_start=ec.pos)
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
            ec.add_segment("notes", note_prompt_ids + [note_token_id] * note_slots, grad=train)
            ec.rebuild(keep_steps, note_cap)
    if train and window_losses:
        (sum(window_losses) / max(n, 1) * scale).backward()
    return losses, actions_ok


class SafeRebuildCache(EpisodeCache):
    """EpisodeCache with a keep_steps=0-safe rebuild (list[-0:] is the whole list)."""

    def rebuild(self, keep_steps, note_cap):
        keep_ranges = [seg for seg in self.segments if seg[0] == "header"]
        notes = [seg for seg in self.segments if seg[0] == "notes"]
        keep_ranges += notes[-note_cap:]
        if keep_steps > 0:
            keep_ranges += [seg for seg in self.segments if seg[0] == "step"][-keep_steps:]
        keep_ranges.sort(key=lambda seg: seg[1])
        idx = [i for seg in keep_ranges for i in range(seg[1], seg[2])]
        fresh = DynamicCache()
        for i, layer in enumerate(self.cache.layers):
            fresh.update(layer.keys[:, :, idx, :], layer.values[:, :, idx, :], i)
        self.cache = fresh
        new_segments, col = [], 0
        for kind, s, e in keep_ranges:
            new_segments.append([kind, col, col + (e - s)])
            col += e - s
        self.segments = new_segments


def episode_losses_c5(model, tokenizer, task, ep, device, max_length, train=True, scale=1.0,
                      note_slots=16, note_window=12, note_cap=4, keep_steps=2,
                      note_prompt_ids=None, note_token_id=None, max_train_steps=None):
    """C5: unified compression. Identical to c4's loop but the persistent memory
    after each boundary is header + note slots only (keep_steps forced to 0)."""
    header = [TASK_HEADERS[task], f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]
    n = len(ep["steps"])
    ec = SafeRebuildCache(model, tokenizer, device, None)
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
        logits = score_leaves(model, leaf_ids, ec.cache, device, grad=train, pos_start=ec.pos)
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
            ec.add_segment("notes", note_prompt_ids + [note_token_id] * note_slots, grad=train)
            ec.rebuild(0, note_cap)
    if train and window_losses:
        (sum(window_losses) / max(n, 1) * scale).backward()
    return losses, actions_ok


class EpisodeCacheC6(EpisodeCache):
    """EpisodeCache that captures the last segment's final hidden state and can
    append externally computed KV columns (no token forward)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last_hidden = None

    def append(self, ids, grad=False):
        total = (self.cache.get_seq_length() if self.cache is not None else 0) + len(ids)
        mask = torch.ones((1, total), dtype=torch.long, device=self.device)
        pos = torch.arange(self.pos, self.pos + len(ids), device=self.device).unsqueeze(0)
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = self.model.backbone(input_ids=tokens, attention_mask=mask,
                                      position_ids=pos, past_key_values=self.cache,
                                      use_cache=True)
        self.cache = out.past_key_values
        self.last_hidden = out.last_hidden_state[0, -1]
        start = self.pos
        self.pos += len(ids)
        return start, self.pos

    def append_kv(self, keys, values, n):
        """Append n precomputed KV columns per layer; advances absolute positions."""
        col_start = self.length()
        for i in range(len(self.cache.layers)):
            self.cache.update(keys[i].contiguous(), values[i].contiguous(), i)
        self.segments.append(["notes", col_start, col_start + n])
        self.pos += n


class LatentNoteWriter(nn.Module):
    """Learned write head. Base slot KV comes from forwarding free continuous
    slot embeddings over the cache (in-distribution, produced by the backbone);
    a gated low-rank delta is projected directly from a hidden state, bypassing
    the tokenizer. Gates/slot scales start at zero => exact c4-style start."""

    def __init__(self, backbone_cfg, note_slots, rank=64):
        super().__init__()
        h = backbone_cfg.hidden_size
        self.note_slots = note_slots
        self.note_embed = nn.Parameter(torch.randn(note_slots, h) * 0.02)
        self.h_proj = nn.Linear(h, rank)
        nl = backbone_cfg.num_hidden_layers
        self.k_out = nn.ModuleList([nn.Linear(rank, backbone_cfg.num_key_value_heads * backbone_cfg.head_dim)
                                    for _ in range(nl)])
        self.v_out = nn.ModuleList([nn.Linear(rank, backbone_cfg.num_key_value_heads * backbone_cfg.head_dim)
                                    for _ in range(nl)])
        self.slot_scale = nn.Parameter(torch.zeros(note_slots))
        self.gate = nn.Parameter(torch.zeros(nl, 2))


def write_latent_notes(model, writer, ec, device, grad=True):
    slots = writer.note_slots
    ctx = torch.enable_grad() if grad else torch.no_grad()
    view = DynamicCache()
    for i, layer in enumerate(ec.cache.layers):
        view.update(layer.keys, layer.values, i)
    pos_ids = torch.arange(ec.pos, ec.pos + slots, device=device).unsqueeze(0)
    with ctx:
        out = model.backbone(inputs_embeds=writer.note_embed.unsqueeze(0),
                             position_ids=pos_ids,
                             attention_mask=torch.ones((1, ec.length() + slots), dtype=torch.long, device=device),
                             past_key_values=view, use_cache=True)
        z = writer.h_proj(ec.last_hidden)
        keys, values = [], []
        for i in range(len(out.past_key_values.layers)):
            layer = out.past_key_values.layers[i]
            bK, bV = layer.keys[:, :, -slots:, :], layer.values[:, :, -slots:, :]
            dK = writer.k_out[i](z).view(1, bK.shape[1], 1, bK.shape[3]).to(bK.dtype)
            dV = writer.v_out[i](z).view(1, bV.shape[1], 1, bV.shape[3]).to(bV.dtype)
            g = writer.gate[i]
            ss = writer.slot_scale.view(1, 1, slots, 1).to(bK.dtype)
            keys.append(bK + (g[0].to(bK.dtype) * ss) * dK)
            values.append(bV + (g[1].to(bV.dtype) * ss) * dV)
    ec.append_kv(keys, values, slots)


def episode_losses_c6(model, tokenizer, task, ep, device, writer, max_length, train=True, scale=1.0,
                      note_slots=16, note_window=12, note_cap=4, keep_steps=2,
                      max_train_steps=None):
    """C6: latent write head. Same loop as c4 (two-tier, keep_steps raw recent
    steps) but notes are written by the learned head, not by token forwarding."""
    header = [TASK_HEADERS[task], f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]
    n = len(ep["steps"])
    ec = EpisodeCacheC6(model, tokenizer, device, None)
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
        logits = score_leaves(model, leaf_ids, ec.cache, device, grad=train, pos_start=ec.pos)
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
            write_latent_notes(model, writer, ec, device, grad=train)
            ec.rebuild(keep_steps, note_cap)
    if train and window_losses:
        (sum(window_losses) / max(n, 1) * scale).backward()
    return losses, actions_ok


def episode_losses_c7(model, tokenizer, task, ep, device, writer, max_length, train=True, scale=1.0,
                      note_slots=16, note_window=12, note_cap=4,
                      max_train_steps=None):
    """C7: fully-unified latent memory -- single persistent tier (compressed
    slots only, keep_steps=0 as in c5) written by the latent write head (as in
    c6). The only remaining token pathway is perception: raw step KV lives
    transiently inside the window and is evicted at every boundary."""
    header = [TASK_HEADERS[task], f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]
    n = len(ep["steps"])
    ec = EpisodeCacheC6(model, tokenizer, device, None)
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
        logits = score_leaves(model, leaf_ids, ec.cache, device, grad=train, pos_start=ec.pos)
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
            write_latent_notes(model, writer, ec, device, grad=train)
            ec.rebuild(0, note_cap)
    if train and window_losses:
        (sum(window_losses) / max(n, 1) * scale).backward()
    return losses, actions_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["c3", "c4", "c5", "c6", "c7"], required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--checkpoint", default="checkpoints/NanoJev-unified")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--train-episodes", type=int, default=0)
    p.add_argument("--episodes-per-update", type=int, default=1)
    p.add_argument("--bptt-window", type=int, default=8)
    p.add_argument("--note-slots", type=int, default=16)
    p.add_argument("--note-window", type=int, default=12)
    p.add_argument("--note-cap", type=int, default=4)
    p.add_argument("--keep-steps", type=int, default=2)
    p.add_argument("--max-train-steps", type=int, default=60)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--mem-fraction", type=float, default=0.9)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    runtime = load_decision_model(a.checkpoint, mem_fraction=a.mem_fraction)
    model, tokenizer = runtime.model, runtime.tokenizer
    device = next(model.parameters()).device
    max_length = runtime.limit
    note_prompt_ids = tokenizer.encode(NOTE_PROMPT, add_special_tokens=False)
    note_token_id = tokenizer.encode("|", add_special_tokens=False)[0]
    writer = None
    if a.arm in ("c6", "c7"):
        # attach as a submodule: writer params enter the head optimizer group
        # and are saved inside the bundle's state_dict (writer.* keys)
        writer = LatentNoteWriter(model.backbone.config, a.note_slots).to(device)
        with torch.no_grad():
            # warm-start slot embeddings from the pretrained "|" token so the
            # zero-gate start matches c5-style notes (random slots otherwise
            # need many steps to enter the backbone's embedding distribution)
            base = model.backbone.get_input_embeddings().weight[note_token_id]
            writer.note_embed.copy_(base.unsqueeze(0)
                                    + 0.01 * torch.randn_like(writer.note_embed))
        model.writer = writer

    train_eps = load_episodes(a.train, a.train_episodes)
    dev_eps = load_episodes(a.dev)[: a.eval_episodes]
    print(f"arm={a.arm} task={a.task} train episodes={len(train_eps)} dev={len(dev_eps)}", flush=True)

    def run_episode(ep, train, scale):
        kw = dict(train=train, scale=scale, max_length=max_length)
        if a.arm == "c3":
            return episode_losses_c3(model, tokenizer, a.task, ep, device,
                                     bptt_window=a.bptt_window,
                                     max_train_steps=a.max_train_steps, **kw)
        if a.arm == "c5":
            return episode_losses_c5(model, tokenizer, a.task, ep, device,
                                     note_slots=a.note_slots, note_window=a.note_window,
                                     note_cap=a.note_cap,
                                     note_prompt_ids=note_prompt_ids, note_token_id=note_token_id,
                                     max_train_steps=a.max_train_steps, **kw)
        if a.arm == "c6":
            return episode_losses_c6(model, tokenizer, a.task, ep, device, writer,
                                     note_slots=a.note_slots, note_window=a.note_window,
                                     note_cap=a.note_cap, keep_steps=a.keep_steps,
                                     max_train_steps=a.max_train_steps, **kw)
        if a.arm == "c7":
            return episode_losses_c7(model, tokenizer, a.task, ep, device, writer,
                                     note_slots=a.note_slots, note_window=a.note_window,
                                     note_cap=a.note_cap,
                                     max_train_steps=a.max_train_steps, **kw)
        return episode_losses_c4(model, tokenizer, a.task, ep, device,
                                 note_slots=a.note_slots, note_window=a.note_window,
                                 note_cap=a.note_cap, keep_steps=a.keep_steps,
                                 note_prompt_ids=note_prompt_ids, note_token_id=note_token_id,
                                 max_train_steps=a.max_train_steps, **kw)

    def dev_eval():
        model.eval()
        total, n, ok = 0.0, 0, 0
        with torch.no_grad():
            for ep in dev_eps:
                losses, good = run_episode(ep, False, 1.0)
                total += sum(losses)
                n += len(losses)
                ok += good
        model.train()
        return total / max(n, 1), ok / max(n, 1)

    body = list(model.backbone.parameters())
    head = [p_ for name, p_ in model.named_parameters() if not name.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.backbone_lr}, {"params": head, "lr": a.head_lr}],
                            weight_decay=0.01)
    ce0, acc0 = dev_eval()
    print(f"initial dev CE {ce0:.4f} teacher-action acc {acc0:.3f}", flush=True)
    save_bundle(model, tokenizer, a.checkpoint, a.out)
    best, best_step, logs = ce0, 0, []
    order = list(range(len(train_eps)))
    random.Random(a.seed).shuffle(order)
    started, pointer, step_losses = time.time(), 0, []
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.episodes_per_update):
            ep = train_eps[order[pointer % len(order)]]
            pointer += 1
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses, _ = run_episode(ep, True, 1.0 / a.episodes_per_update)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"step {step}: OOM on episode {pointer}, skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            torch.cuda.empty_cache()
            if losses:
                step_losses.append(sum(losses) / len(losses))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 and step_losses:
            print(f"step {step} loss {sum(step_losses)/len(step_losses):.4f} "
                  f"elapsed {time.time()-started:.0f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
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
        {"arm": a.arm, "task": a.task, "config": vars(a), "initial_dev_ce": ce0,
         "initial_dev_acc": acc0, "best_step": best_step, "best_dev_ce": best, "logs": logs}, indent=2))
    print(json.dumps({"done": a.out, "best_step": best_step, "best_dev_ce": best}))


if __name__ == "__main__":
    main()
