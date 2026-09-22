"""Shared loaders/text templates for the unified benchmark episode format.

Unified episode JSONL (one object per line):
  {"task": ..., "episode_id": ..., "split": ..., "meta": {...}, "success": bool,
   "n_steps": int, "steps": [{"t": i, "obs": str, "candidates": {key: desc},
                              "action": key, "event": str}, ...]}

Grid files (maze*/snake*) use the same step layout with legacy top-level keys
(size/seed/epsilon), which load_episodes tolerates.
"""
import json
from pathlib import Path

from jet.vendor.unified_game_pipeline import POLICY_QUESTION

TASK_HEADERS = {
    "maze": "Maze navigation episode record. Each step lists the visible observation, the asked question, the decision taken, and its result.\n",
    "snake": "Snake game episode record. Each step lists the visible observation, the asked question, the decision taken, and its result.\n",
    "pokemon": "Pokemon battle episode record. Each step lists the visible battle state, the asked question, the chosen action, and its result.\n",
    "webshop": "Web shopping task record. Each step lists the page observation, the asked question, the chosen action, and its result.\n",
    "alfworld": "Text-world household task record. Each step lists the observation, the asked question, the chosen action, and its result.\n",
    "mind2web": "Web navigation task record. Each step lists the page state, the asked question, the chosen action, and its result.\n",
}


def load_episodes(path, max_episodes=0):
    """Load unified-format episodes; drops steps whose action is not a candidate key.

    Splits on "\n" only (not str.splitlines): product/DOM text may contain
    U+2028/U+2029, which are legal inside JSON strings but splitlines() treats
    as line breaks.
    """
    episodes, dropped = [], 0
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        ep = json.loads(line)
        steps = [s for s in ep["steps"] if s.get("action") in (s.get("candidates") or {})
                 and len(s.get("candidates") or {}) >= 1]
        dropped += len(ep["steps"]) - len(steps)
        if steps:
            ep["steps"] = steps
            episodes.append(ep)
        if max_episodes and len(episodes) >= max_episodes:
            break
    if dropped:
        print(f"note: dropped {dropped} steps with action not in candidates", flush=True)
    return episodes


def state_segments(task, state_text):
    header = TASK_HEADERS.get(task, f"{task} episode record.\n")
    return [header, f"State:\n{state_text}\n", f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]


def build_generic_example(tokenizer, task, state_text, candidates, max_length):
    """Choice example with leaves in candidate presentation order (dict order)."""
    prefix = []
    for text in state_segments(task, state_text):
        prefix.extend(tokenizer.encode(text, add_special_tokens=False))
    leaves, keys = [], []
    for key, desc in candidates.items():
        leaf = (f"Candidate:\n{key}: {desc}\nDecision:")
        leaves.append(prefix + tokenizer.encode(leaf, add_special_tokens=False)
                      + [tokenizer.eos_token_id])
        keys.append(key)
    longest = max(map(len, leaves))
    if longest > max_length:
        raise ValueError(f"candidate path {longest} tokens exceeds max_length={max_length}")
    return {"id": None, "type": "choice", "candidate_ids": keys, "leaf_tokens": leaves,
            "prefix_tokens": prefix, "leaf_only": [leaf[len(prefix):] for leaf in leaves]}


def episodes_to_examples(episodes, tokenizer, task, max_length, max_examples=0):
    out, skipped = [], 0
    for ep in episodes:
        for s in ep["steps"]:
            try:
                ex = build_generic_example(tokenizer, task, s["obs"], s["candidates"], max_length)
            except ValueError:
                skipped += 1
                continue
            ex["target"] = ex["candidate_ids"].index(s["action"])
            out.append(ex)
            if max_examples and len(out) >= max_examples:
                return out, skipped
    return out, skipped
