#!/usr/bin/env python3
"""Build the six-task mixed training pool (decision tasks capped at 300 episodes),
shuffled into one train/dev pair for single-model cross-task Jet training."""
import json
import random
from pathlib import Path

B = Path("/data/yangyuming/Long-Jev/data/benchmarks")
OUT = B / "mix"
OUT.mkdir(parents=True, exist_ok=True)
CAPS = {"maze": 400, "snake": 300, "pokemon": 400,     # games: full sets
        "alfworld": 300, "mind2web": 300, "webshop": 300}  # decision: capped


def load(p, cap):
    rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]
    return rows[:cap]


def main():
    rng = random.Random(17)
    train, dev = [], []
    for task, cap in CAPS.items():
        eps = load(B / task / "train.jsonl", cap)
        train += eps
        devs = load(B / task / "dev.jsonl", 10)
        dev += devs
        for e in eps + devs:          # stamp the task label (legacy grid files lack it)
            e["task"] = task
        print(f"{task}: train {len(eps)} dev {len(devs)}")
    rng.shuffle(train)
    rng.shuffle(dev)
    (OUT / "train.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in train), encoding="utf-8")
    (OUT / "dev.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in dev), encoding="utf-8")
    print(f"mix: train {len(train)} episodes -> {OUT/'train.jsonl'}; dev {len(dev)}")


if __name__ == "__main__":
    main()
