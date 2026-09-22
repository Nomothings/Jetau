"""Generate Pokemon battle episode splits for the memory benchmark.

Unified episode JSONL format (one episode object per line):
  {"task":"pokemon","episode_id":"pokemon-<split>-<idx>","split":...,"meta":{...},
   "success":bool,"n_steps":N,"steps":[{"t":1,"obs":...,"candidates":{...},
   "action":<candidate key>,"event":...}]}

Splits: train 400 / dev 60 / test 100 battles; seed ranges are disjoint
(100000+, 200000+, 300000+). Teams are drawn from the 17-species roster by the
battle seed itself, so every battle is a pure function of (seed, epsilon).
"""
import argparse
import json
from pathlib import Path

from jet.pokemon_env import run_episode

SPLITS = {
    "train": (400, range(100000, 100400), [0.15, 0.25]),
    "dev": (60, range(200000, 200060), [0.05]),
    "test": (100, range(300000, 300100), [0.05]),
}

TOP_KEYS = {"task", "episode_id", "split", "meta", "success", "n_steps", "steps"}
STEP_KEYS = {"t", "obs", "candidates", "action", "event"}


def validate_episode(ep):
    """Hard schema assertions for the unified format."""
    assert set(ep) == TOP_KEYS, f"bad top-level keys: {sorted(ep)}"
    assert ep["task"] == "pokemon"
    assert ep["episode_id"].startswith(f"pokemon-{ep['split']}-")
    assert isinstance(ep["success"], bool)
    assert isinstance(ep["n_steps"], int) and ep["n_steps"] == len(ep["steps"]) >= 1
    assert isinstance(ep["meta"], dict) and isinstance(ep["meta"]["seed"], int)
    for k, s in enumerate(ep["steps"]):
        assert set(s) == STEP_KEYS, f"step {k} bad keys: {sorted(s)}"
        assert s["t"] == k + 1
        assert isinstance(s["obs"], str) and 0 < len(s["obs"]) <= 3500, \
            f"obs length {len(s['obs'])}"
        assert isinstance(s["event"], str) and 0 < len(s["event"]) <= 200, \
            f"event length {len(s['event'])}: {s['event']!r}"
        cands = s["candidates"]
        assert isinstance(cands, dict) and 3 <= len(cands) <= 8, \
            f"candidate count {len(cands)}"
        assert len(cands) <= 10
        assert all(isinstance(v, str) and v for v in cands.values())
        assert s["action"] in cands, f"action {s['action']!r} not in candidates"


def generate(split, n, seeds, eps_mix, out_path):
    rows, wins = [], 0
    for i, seed in enumerate(seeds):
        eps = eps_mix[i % len(eps_mix)]
        raw = run_episode(seed, epsilon=eps)
        ep = {"task": raw["task"], "episode_id": f"pokemon-{split}-{i:04d}",
              "split": split, "meta": raw["meta"], "success": raw["success"],
              "n_steps": raw["n_steps"], "steps": raw["steps"]}
        validate_episode(ep)
        rows.append(ep)
        wins += ep["success"]
    Path(out_path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    lens = [r["n_steps"] for r in rows]
    cands = [len(s["candidates"]) for r in rows for s in r["steps"]]
    max_obs = max(len(s["obs"]) for r in rows for s in r["steps"])
    max_event = max(len(s["event"]) for r in rows for s in r["steps"])
    print(f"{split}: {len(rows)} episodes (seeds {seeds[0]}..{seeds[-1]}), "
          f"teacher wins {wins}/{len(rows)} ({100.0 * wins / len(rows):.1f}%), "
          f"steps mean {sum(lens) / len(lens):.1f} min {min(lens)} max {max(lens)}, "
          f"candidates mean {sum(cands) / len(cands):.2f} min {min(cands)} max {max(cands)}, "
          f"max obs {max_obs} chars, max event {max_event} chars")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    for split, (n, seeds, eps_mix) in SPLITS.items():
        assert len(list(seeds)) == n
        assert not (set(seeds) & seen), "seed ranges must be disjoint across splits"
        seen |= set(seeds)
        generate(split, n, seeds, eps_mix, out / f"{split}.jsonl")


if __name__ == "__main__":
    main()
