#!/usr/bin/env python3
"""Policies for closed-loop game evaluation.

Three strategies sharing one interface `act(obs, candidates, feedback) -> key`:
  nomem     -- step-local scoring, no state (released NanoJev)
  promptmem -- sliding-window text history in the prompt (training-free)
  jet       -- streaming latent memory with the trained write head (bundle)
"""
from pathlib import Path

import torch

from jet.bench_common import POLICY_QUESTION, TASK_HEADERS, build_generic_example
from jet.common import load_decision_model, leaf_text, tokenize_segments
from jet.latent_state import LatentStateCache, LatentStateWriter, write_latent_state
from jet.train_jet_bs import score_leaves_bs


class BasePolicy:
    def __init__(self, checkpoint, task, mem_fraction=0.4):
        self.runtime = load_decision_model(checkpoint, mem_fraction=mem_fraction)
        self.model, self.tok = self.runtime.model, self.runtime.tokenizer
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.task = task
        self.pad = self.tok.pad_token_id

    def _score_flat(self, state_text, candidates):
        ex = build_generic_example(self.tok, self.task, state_text, candidates,
                                   self.runtime.limit)
        with torch.no_grad():
            logits, _ = self.model([ex], self.pad)
        return logits[0], ex["candidate_ids"]


class NoMemPolicy(BasePolicy):
    name = "nomem"

    def act(self, obs, candidates, feedback=None):
        row, keys = self._score_flat(obs, candidates)
        return keys[int(row[:len(keys)].argmax().item())]


class PromptMemPolicy(BasePolicy):
    name = "promptmem"
    MAX_HIST_CHARS = 800

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.hist = []

    def act(self, obs, candidates, feedback=None):
        state = ""
        if self.hist:
            state = "Recent steps (oldest first):\n" + "\n".join(self.hist) + "\n"
        state += f"Current observation:\n{obs}"
        while True:
            try:
                ex = build_generic_example(self.tok, self.task, state, candidates,
                                           self.runtime.limit)
                break
            except ValueError:
                if self.hist:
                    self.hist.pop(0)
                else:
                    obs = obs[: max(600, len(obs) - 400)]
                    state = f"Current observation:\n{obs}"
        with torch.no_grad():
            logits, _ = self.model([ex], self.pad)
        keys = ex["candidate_ids"]
        return keys[int(logits[0][:len(keys)].argmax().item())]

    def remember(self, obs, action, feedback):
        self.hist.append(f"Obs: {obs[:self.MAX_HIST_CHARS]}\nDid: {action} -> {feedback}")

    def reset(self):
        self.hist = []


class JetPolicy(BasePolicy):
    name = "jet"

    def __init__(self, checkpoint, task, note_slots=16, note_window=6, note_cap=4,
                 mem_fraction=0.4):
        import os, shutil, tempfile
        from safetensors.torch import load_file, save_file
        src = Path(checkpoint)
        sd = load_file(str(src / "best.safetensors"))
        writer_sd = {k[len("writer."):]: v for k, v in sd.items() if k.startswith("writer.")}
        tmp = None
        ckpt = str(src)
        if writer_sd:
            # DecisionPredictor loads strictly; route model-only weights through a
            # temp bundle, then attach the writer separately.
            tmp = Path(tempfile.mkdtemp(prefix="jetpol_", dir=os.environ.get("JET_TMPDIR") or None))
            shutil.copy(src / "config.json", tmp / "config.json")
            shutil.copytree(src / "tokenizer", tmp / "tokenizer")
            shutil.copytree(src / "backbone_config", tmp / "backbone_config")
            save_file({k: v for k, v in sd.items() if not k.startswith("writer.")},
                      tmp / "best.safetensors")
            ckpt = str(tmp)
        super().__init__(ckpt, task, mem_fraction)
        if not writer_sd:
            raise ValueError("Jeτ checkpoint must include writer weights")
        self.writer = LatentStateWriter(self.model.backbone.config, note_slots).to(self.device)
        self.writer.load_state_dict(writer_sd)
        self.model.writer = self.writer
        shutil.rmtree(tmp, ignore_errors=True)
        self.note_slots, self.note_window, self.note_cap = note_slots, note_window, note_cap
        self.reset()

    def reset(self):
        self.steps_in_window = 0
        header = [TASK_HEADERS[self.task], f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]
        self.ec = LatentStateCache(self.model, self.tok, self.device)
        self.ec.add_segment("header", tokenize_segments(self.tok, header), grad=False)

    def act(self, obs, candidates, feedback=None):
        seg = tokenize_segments(self.tok, [f"State:\n{obs}\n"])
        if self.ec.length() + len(seg) + 64 > self.runtime.limit:
            self.ec.rebuild(self.note_cap)
        self.ec.add_segment("step", seg, grad=False)
        keys = list(candidates)
        leaf_ids = [self.tok.encode(leaf_text(k, candidates[k]), add_special_tokens=False)
                    + [self.tok.eos_token_id] for k in keys]
        with torch.no_grad():
            z = score_leaves_bs(self.model, leaf_ids, self.ec.cache, self.device,
                                grad=False, pos_start=self.ec.pos)
        return keys[int(z.argmax().item())]

    def remember(self, obs, action, feedback):
        outcome = f"Decision taken: {action}. Result: {feedback}"
        self.ec.add_segment("step", tokenize_segments(self.tok, [outcome]), grad=False)
        self.steps_in_window += 1
        if self.steps_in_window >= self.note_window:
            with torch.no_grad():
                write_latent_state(self.model, self.writer, self.ec, self.device, grad=False)
            self.ec.rebuild(self.note_cap)
            self.steps_in_window = 0
