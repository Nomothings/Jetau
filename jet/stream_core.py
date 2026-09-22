"""Streaming-cache primitives shared by all Jet trainers and evaluators.

Function bodies extracted verbatim from the Long-Jev memexp trainers
(train_c3.py / train_c4.py), which are not part of the Jet package:

  forward_segment / detach_cache / score_leaves   <- train_c3.py
  NOTE_PROMPT / EpisodeCache                       <- train_c4.py
"""
import torch  # noqa: F401  (imported for callers of this module)
from transformers import DynamicCache

NOTE_PROMPT = "\nMemory update after recent steps:\n"


def forward_segment(model, ids, cache, device, grad=False):
    """Append `ids` to the persistent cache (or start it); cache carries the
    graph when grad=True. Leaf scoring uses `score_leaves`, never this."""
    total = (cache.get_seq_length() if cache is not None else 0) + len(ids)
    mask = torch.ones((1, total), dtype=torch.long, device=device)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model.backbone(input_ids=tokens, attention_mask=mask,
                             past_key_values=cache, use_cache=True)
    return out


def detach_cache(cache):
    fresh = DynamicCache()
    for i, layer in enumerate(cache.layers):
        fresh.update(layer.keys.detach(), layer.values.detach(), i)
    return fresh


def score_leaves(model, leaf_ids, cache, device, grad=True, pos_start=None):
    """Score candidate leaves against a clean zero-copy view of the cache.

    pos_start: absolute position of the first leaf token. Required when the
    cache has position gaps (tiered memory rebuilds); None keeps the default
    contiguous positions (plain C3)."""
    past_len = cache.get_seq_length()
    hiddens = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    for ids in leaf_ids:
        view = DynamicCache()
        for i, layer in enumerate(cache.layers):
            view.update(layer.keys, layer.values, i)
        kwargs = {}
        if pos_start is not None:
            kwargs["position_ids"] = torch.arange(
                pos_start, pos_start + len(ids), device=device).unsqueeze(0)
        with ctx:
            out = model.backbone(
                input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                attention_mask=torch.ones((1, past_len + len(ids)), dtype=torch.long, device=device),
                past_key_values=view, use_cache=False, **kwargs)
        hiddens.append(out.last_hidden_state[0, -1])
    h = model.norm(torch.stack(hiddens).unsqueeze(0))
    z = model.scalar(h).squeeze(-1).float()
    if getattr(model, "set_head", None) == "attention":
        k = h.shape[1]
        valid = torch.ones((1, k), dtype=torch.bool, device=device)
        log_k = valid.sum(-1).float().log()[:, None, None].expand(-1, k, 1)
        u = model.set_project(torch.cat([h, log_k.to(h.dtype)], dim=-1))
        mixed, _ = model.set_attention(u, u, u, key_padding_mask=~valid, need_weights=False)
        z = z + model.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
    return z[0]


class EpisodeCache:
    """Streaming cache with absolute-position bookkeeping and tiered rebuild."""

    def __init__(self, model, tokenizer, device, note_ids, keep_kinds=("header", "notes")):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.note_ids = note_ids
        self.cache = None
        self.pos = 0            # next absolute position
        self.segments = []      # list of (kind, start, end) in cache columns

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
        start = self.pos
        self.pos += len(ids)
        return start, self.pos

    def add_segment(self, kind, ids, grad=False):
        col_start = self.length()          # segment range in cache columns
        self.append(ids, grad=grad)        # advances absolute positions only
        self.segments.append([kind, col_start, col_start + len(ids)])

    def rebuild(self, keep_steps, note_cap):
        keep_ranges = [seg for seg in self.segments if seg[0] == "header"]
        notes = [seg for seg in self.segments if seg[0] == "notes"]
        keep_ranges += notes[-note_cap:]
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

    def length(self):
        return self.cache.get_seq_length() if self.cache is not None else 0
