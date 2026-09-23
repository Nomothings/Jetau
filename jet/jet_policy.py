#!/usr/bin/env python3
"""Policies for closed-loop game evaluation.

Three strategies sharing one interface `act(obs, candidates, feedback) -> key`:
  nomem     -- step-local scoring, no state (released NanoJev)
  promptmem -- sliding-window text history in the prompt (training-free)
  jet       -- streaming latent memory with the trained write head (bundle)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jet.bench_common import TASK_HEADERS, build_generic_example
from jet.common import load_decision_model, leaf_text, tokenize_segments
from jet.stream_core import score_leaves
from jet.stream_core import NOTE_PROMPT, EpisodeCache
from jet.train_mem_generic import EpisodeCacheC6, LatentNoteWriter, write_latent_notes
from jet.stream_core import detach_cache
from transformers import DynamicCache
from jet.vendor.unified_game_pipeline import POLICY_QUESTION


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


class JetPolicy(BasePolicy):
    name = "jet"

    def __init__(self, checkpoint, task, note_slots=16, note_window=6, note_cap=4,
                 note_window_override=None, mem_fraction=0.4):
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
        if writer_sd:
            self.writer = LatentNoteWriter(self.model.backbone.config, note_slots).to(self.device)
            self.writer.load_state_dict(writer_sd)
            self.model.writer = self.writer
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            self.writer = None
        self.note_slots, self.note_window, self.note_cap = note_slots, (note_window_override or note_window), note_cap
        self.steps_in_window = 0
        header = [TASK_HEADERS[task], f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]
        self.ec = EpisodeCacheC6(self.model, self.tok, self.device, None)
        self.ec.add_segment("header", tokenize_segments(self.tok, header), grad=False)

    def act(self, obs, candidates, feedback=None):
        seg = tokenize_segments(self.tok, [f"State:\n{obs}\n"])
        if self.ec.length() + len(seg) + 64 > self.runtime.limit:
            self.ec.rebuild(0, self.note_cap)  # emergency compress: keep slots only
        self.ec.add_segment("step", seg, grad=False)
        keys = list(candidates)
        leaf_ids = [self.tok.encode(leaf_text(k, candidates[k]), add_special_tokens=False)
                    + [self.tok.eos_token_id] for k in keys]
        with torch.no_grad():
            # batched leaves: reuse the bs scorer logic inline (read-only view)
            past_len = self.ec.cache.get_seq_length()
            K, maxlen = len(leaf_ids), max(len(x) for x in leaf_ids)
            tokens = torch.full((K, maxlen), leaf_ids[0][-1], dtype=torch.long, device=self.device)
            attn = torch.ones((K, past_len + maxlen), dtype=torch.long, device=self.device)
            for i, ids in enumerate(leaf_ids):
                tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attn[i, past_len + len(ids):] = 0
            pos = torch.arange(self.ec.pos, self.ec.pos + maxlen, device=self.device).unsqueeze(0).expand(K, maxlen)
            from jet.train_jet_bs import BroadcastNoStoreCache
            view = BroadcastNoStoreCache(self.ec.cache, K)
            out = self.model.backbone(input_ids=tokens, attention_mask=attn, position_ids=pos,
                                      past_key_values=view, use_cache=False)
            last = torch.stack([out.last_hidden_state[i, len(ids) - 1]
                                for i, ids in enumerate(leaf_ids)]).unsqueeze(0)
            h = self.model.norm(last)
            z = self.model.scalar(h).squeeze(-1).float()
            if getattr(self.model, "set_head", None) == "attention":
                valid = torch.ones((1, K), dtype=torch.bool, device=self.device)
                log_k = valid.sum(-1).float().log()[:, None, None].expand(-1, K, 1)
                u = self.model.set_project(torch.cat([h, log_k.to(h.dtype)], dim=-1))
                mixed, _ = self.model.set_attention(u, u, u, key_padding_mask=~valid, need_weights=False)
                z = z + self.model.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        return keys[int(z.argmax().item())]

    def remember(self, obs, action, feedback):
        outcome = f"Decision taken: {action}. Result: {feedback}"
        self.ec.add_segment("step", tokenize_segments(self.tok, [outcome]), grad=False)
        self.steps_in_window += 1
        if self.steps_in_window >= self.note_window:
            if self.writer is not None:
                with torch.no_grad():
                    write_latent_notes(self.model, self.writer, self.ec, self.device, grad=False)
            else:
                ids = self.tok.encode(NOTE_PROMPT, add_special_tokens=False) + \
                      [self.tok.encode("|", add_special_tokens=False)[0]] * self.note_slots
                self.ec.add_segment("notes", ids, grad=False)
            self.ec.rebuild(0, self.note_cap)
            self.steps_in_window = 0
