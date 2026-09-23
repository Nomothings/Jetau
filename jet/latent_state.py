"""Persistent latent state shared by Jeτ training and inference."""

import torch
import torch.nn as nn
from transformers import DynamicCache

def step_outcome(action, event):
    return f"Decision taken: {action}. Result: {event}\n"


def detach_cache(cache):
    fresh = DynamicCache()
    for layer, entry in enumerate(cache.layers):
        fresh.update(entry.keys.detach(), entry.values.detach(), layer)
    return fresh


class LatentStateCache:
    """Keep the task header and bounded latent state across write windows."""

    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.cache = None
        self.pos = 0
        self.segments = []
        self.last_hidden = None

    def length(self):
        return self.cache.get_seq_length() if self.cache is not None else 0

    def add_segment(self, kind, ids, grad=False):
        start = self.length()
        self.append(ids, grad=grad)
        self.segments.append([kind, start, start + len(ids)])

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

    def append_kv(self, keys, values, count):
        start = self.length()
        for layer in range(len(self.cache.layers)):
            self.cache.update(keys[layer].contiguous(), values[layer].contiguous(), layer)
        self.segments.append(["notes", start, start + count])
        self.pos += count

    def rebuild(self, note_cap):
        # Only the header and the latest latent writes persist across windows.
        keep = [segment for segment in self.segments if segment[0] == "header"]
        notes = [segment for segment in self.segments if segment[0] == "notes"]
        keep += notes[-note_cap:]
        keep.sort(key=lambda segment: segment[1])
        indices = [i for _, start, end in keep for i in range(start, end)]
        fresh = DynamicCache()
        for layer, entry in enumerate(self.cache.layers):
            fresh.update(entry.keys[:, :, indices, :], entry.values[:, :, indices, :], layer)
        self.cache = fresh
        self.segments = []
        offset = 0
        for kind, start, end in keep:
            self.segments.append([kind, offset, offset + end - start])
            offset += end - start


class LatentStateWriter(nn.Module):
    """Write continuous state embeddings and a gated KV update."""

    def __init__(self, backbone_cfg, note_slots, rank=64):
        super().__init__()
        hidden = backbone_cfg.hidden_size
        self.note_slots = note_slots
        self.note_embed = nn.Parameter(torch.randn(note_slots, hidden) * 0.02)
        self.h_proj = nn.Linear(hidden, rank)
        layers = backbone_cfg.num_hidden_layers
        width = backbone_cfg.num_key_value_heads * backbone_cfg.head_dim
        self.k_out = nn.ModuleList([nn.Linear(rank, width) for _ in range(layers)])
        self.v_out = nn.ModuleList([nn.Linear(rank, width) for _ in range(layers)])
        self.slot_scale = nn.Parameter(torch.zeros(note_slots))
        self.gate = nn.Parameter(torch.zeros(layers, 2))


def write_latent_state(model, writer, cache, device, grad=True):
    slots = writer.note_slots
    ctx = torch.enable_grad() if grad else torch.no_grad()
    view = DynamicCache()
    for layer, entry in enumerate(cache.cache.layers):
        view.update(entry.keys, entry.values, layer)
    positions = torch.arange(cache.pos, cache.pos + slots, device=device).unsqueeze(0)
    with ctx:
        out = model.backbone(
            inputs_embeds=writer.note_embed.unsqueeze(0),
            position_ids=positions,
            attention_mask=torch.ones((1, cache.length() + slots), dtype=torch.long, device=device),
            past_key_values=view,
            use_cache=True,
        )
        hidden = writer.h_proj(cache.last_hidden)
        keys, values = [], []
        for layer, entry in enumerate(out.past_key_values.layers):
            base_key = entry.keys[:, :, -slots:, :]
            base_value = entry.values[:, :, -slots:, :]
            delta_key = writer.k_out[layer](hidden).view(1, base_key.shape[1], 1, base_key.shape[3])
            delta_value = writer.v_out[layer](hidden).view(1, base_value.shape[1], 1, base_value.shape[3])
            gate = writer.gate[layer]
            scale = writer.slot_scale.view(1, 1, slots, 1)
            keys.append(base_key + gate[0].to(base_key.dtype) * scale.to(base_key.dtype) * delta_key.to(base_key.dtype))
            values.append(base_value + gate[1].to(base_value.dtype) * scale.to(base_value.dtype) * delta_value.to(base_value.dtype))
    cache.append_kv(keys, values, slots)
