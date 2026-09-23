#!/usr/bin/env python3
"""Teacher-forced step-accuracy evaluation for memory-arm bundles (c3-c7).

Loads a bundle saved by train_mem_generic/train_jet (backbone+head+writer),
streams unified-format episodes with no_grad through the given arm, and
reports mean CE and top-1 step accuracy on the test file.
"""
import argparse
import json
from pathlib import Path

import torch

from jet.bench_common import TASK_HEADERS, load_episodes
from jet.common import leaf_text, load_decision_model, tokenize_segments
from jet.train_mem_generic import (LatentNoteWriter, episode_losses_c3, episode_losses_c4,
                                   episode_losses_c5, episode_losses_c6, episode_losses_c7)
from jet.stream_core import NOTE_PROMPT
from transformers import DynamicCache  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["c3", "c4", "c5", "c6", "c7"], required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--limit-episodes", type=int, default=200)
    p.add_argument("--note-slots", type=int, default=16)
    p.add_argument("--note-window", type=int, default=6)
    p.add_argument("--note-cap", type=int, default=4)
    p.add_argument("--keep-steps", type=int, default=2)
    p.add_argument("--max-train-steps", type=int, default=0, help="0 = full episode")
    p.add_argument("--out", default="")
    p.add_argument("--mem-fraction", type=float, default=0.35)
    a = p.parse_args()

    import os, shutil, tempfile
    from safetensors.torch import load_file, save_file
    src = Path(a.checkpoint)
    sd = load_file(str(src / "best.safetensors"))
    writer_sd = {k[len("writer."):]: v for k, v in sd.items() if k.startswith("writer.")}
    if writer_sd:
        # DecisionPredictor loads strictly; route the model-only weights through
        # a temp bundle and load the writer separately.
        # JET_TMPDIR can point to a volume with room for a large bundle.
        tmp = Path(tempfile.mkdtemp(prefix="evalmem_", dir=os.environ.get("JET_TMPDIR") or None))
        shutil.copy(src / "config.json", tmp / "config.json")
        shutil.copytree(src / "tokenizer", tmp / "tokenizer")
        shutil.copytree(src / "backbone_config", tmp / "backbone_config")
        save_file({k: v for k, v in sd.items() if not k.startswith("writer.")},
                  tmp / "best.safetensors")
        ckpt = str(tmp)
    else:
        ckpt = str(src)
    runtime = load_decision_model(ckpt, mem_fraction=a.mem_fraction)
    model, tokenizer = runtime.model, runtime.tokenizer
    device = next(model.parameters()).device
    if a.arm in ("c6", "c7"):
        assert writer_sd, "c6/c7 bundle must contain writer weights"
        writer = LatentNoteWriter(model.backbone.config, a.note_slots).to(device)
        writer.load_state_dict(writer_sd)
        model.writer = writer
    model.eval()

    eps = load_episodes(a.test, a.limit_episodes)
    note_prompt_ids = tokenizer.encode(NOTE_PROMPT, add_special_tokens=False)
    note_token_id = tokenizer.encode("|", add_special_tokens=False)[0]
    total_loss, total_steps, correct = 0.0, 0, 0
    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, ep in enumerate(eps):
            base = dict(max_length=runtime.limit,
                        note_window=a.note_window, note_cap=a.note_cap)
            if a.arm == "c3":
                losses, ok = episode_losses_c3(model, tokenizer, a.task, ep, device, train=False, **base)
            elif a.arm == "c4":
                losses, ok = episode_losses_c4(model, tokenizer, a.task, ep, device, train=False,
                                               keep_steps=a.keep_steps,
                                               note_prompt_ids=note_prompt_ids,
                                               note_token_id=note_token_id, **base)
            elif a.arm == "c5":
                losses, ok = episode_losses_c5(model, tokenizer, a.task, ep, device, train=False,
                                               note_prompt_ids=note_prompt_ids,
                                               note_token_id=note_token_id, **base)
            elif a.arm == "c6":
                losses, ok = episode_losses_c6(model, tokenizer, a.task, ep, device,
                                               writer=model.writer, train=False,
                                               keep_steps=a.keep_steps, **base)
            else:
                losses, ok = episode_losses_c7(model, tokenizer, a.task, ep, device,
                                               writer=model.writer, train=False, **base)
            total_loss += sum(losses)
            total_steps += len(losses)
            correct += ok
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(eps)} episodes, acc so far {correct/max(total_steps,1):.3f}", flush=True)
    wall = time.perf_counter() - t0
    result = {"arm": a.arm, "task": a.task, "checkpoint": a.checkpoint, "test": a.test,
              "n_episodes": len(eps), "n_steps": total_steps,
              "step_acc": correct / max(total_steps, 1), "mean_ce": total_loss / max(total_steps, 1),
              "ms_per_step": wall / max(total_steps, 1) * 1000,
              "steps_per_sec": total_steps / max(wall, 1e-9)}
    if writer_sd:
        shutil.rmtree(tmp, ignore_errors=True)  # free the multi-GB temp bundle
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
