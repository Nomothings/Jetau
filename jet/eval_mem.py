"""Step-level accuracy evaluation for a trained Jeτ checkpoint."""

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from jet.bench_common import load_episodes
from jet.common import load_decision_model
from jet.latent_state import LatentStateWriter
from jet.train_jet_bs import episode_losses_jet_bs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit-episodes", type=int, default=0,
                        help="0 evaluates the full test set")
    parser.add_argument("--note-slots", type=int, default=16)
    parser.add_argument("--note-window", type=int, default=6)
    parser.add_argument("--note-cap", type=int, default=4)
    parser.add_argument("--out", default="")
    parser.add_argument("--mem-fraction", type=float, default=0.35)
    args = parser.parse_args()

    source = Path(args.checkpoint)
    weights = load_file(str(source / "best.safetensors"))
    writer_weights = {key[len("writer."):]: value for key, value in weights.items()
                      if key.startswith("writer.")}
    if not writer_weights:
        raise ValueError("Jeτ checkpoint must include writer weights")

    # The upstream decision loader expects backbone and head weights only.
    with tempfile.TemporaryDirectory(prefix="jet_eval_", dir=os.environ.get("JET_TMPDIR")) as temp:
        bundle = Path(temp)
        shutil.copy(source / "config.json", bundle / "config.json")
        shutil.copytree(source / "tokenizer", bundle / "tokenizer")
        shutil.copytree(source / "backbone_config", bundle / "backbone_config")
        save_file({key: value for key, value in weights.items()
                   if not key.startswith("writer.")}, str(bundle / "best.safetensors"))
        runtime = load_decision_model(str(bundle), mem_fraction=args.mem_fraction)
    model, tokenizer = runtime.model, runtime.tokenizer
    device = next(model.parameters()).device
    writer = LatentStateWriter(model.backbone.config, args.note_slots).to(device)
    writer.load_state_dict(writer_weights)
    model.writer = writer
    model.eval()

    episodes = load_episodes(args.test, args.limit_episodes)
    loss_sum = steps = correct = 0
    started = time.perf_counter()
    with torch.no_grad():
        for episode in episodes:
            losses, right = episode_losses_jet_bs(
                model, tokenizer, args.task, episode, device, runtime.limit, writer,
                train=False, note_slots=args.note_slots,
                note_window=args.note_window, note_cap=args.note_cap,
            )
            loss_sum += sum(losses)
            steps += len(losses)
            correct += right
    elapsed = time.perf_counter() - started
    result = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "test": args.test,
        "n_episodes": len(episodes),
        "n_steps": steps,
        "step_acc": correct / max(steps, 1),
        "mean_ce": loss_sum / max(steps, 1),
        "ms_per_step": 1000 * elapsed / max(steps, 1),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
