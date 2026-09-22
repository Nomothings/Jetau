#!/usr/bin/env python3
"""Assemble the final 4-strategy x 6-scenario comparison table.

Strategies: no-memory NanoJev / prompt-text memory (sliding window) /
Jet-0.6B / Jet-1.7B. Metrics: test step acc, mean CE, inference ms/step,
and (for Jet) final train loss + best dev acc.

Inputs (relative to the repo root): evaluation JSONs written by eval_acc.py /
eval_promptmem.py / eval_mem.py into results/, and train_log.json inside each
fleet bundle under checkpoints/jet/. Output: results/final_table.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
M = ROOT / "checkpoints" / "jet"
TASKS = ["maze", "snake", "pokemon", "alfworld", "mind2web", "webshop"]


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    rows = []
    for t in TASKS:
        nomem = load(R / f"base_nomem_{t}.json")
        pmem = load(R / f"base_promptmem_{t}.json")
        j06 = load(R / f"testmem_{t}_jet06.json")
        j17 = load(R / f"testmem_{t}_jet17.json")
        tl06 = load(M / f"jet06_{t}" / "train_log.json")
        tl17 = load(M / f"jet17_{t}" / "train_log.json")

        def jet_cols(tm, tl):
            if not tm:
                return ["-", "-", "-", "-", "-"]
            loss = "-"
            if tl and tl.get("logs"):
                tr = [x for x in tl["logs"] if x["step"] == tm.get("step")]
                loss = fmt(tl["logs"][-1].get("dev_ce")) if tl["logs"] else "-"
            return [fmt(tm["step_acc"]), loss, fmt(tm.get("ms_per_step"), 1),
                    fmt(tm.get("mean_ce")), fmt((tl or {}).get("best_dev_ce"))]

        rows.append([t,
                     fmt(nomem.get("step_acc")) if nomem else "-",
                     fmt(pmem.get("step_acc")) if pmem else "-",
                     *jet_cols(j06, tl06),
                     *jet_cols(j17, tl17),
                     fmt(nomem.get("ms_per_step"), 1) if nomem else "-",
                     fmt(pmem.get("ms_per_step"), 1) if pmem else "-"])

    lines = ["# Jet: four strategies x six scenarios (test step acc / CE / inference speed)", "",
             "| Task | no-mem acc | prompt-mem acc | Jet-0.6B acc | Jet-0.6B final CE | Jet-0.6B ms/step | Jet-0.6B best dev CE | Jet-1.7B acc | Jet-1.7B final CE | Jet-1.7B ms/step | Jet-1.7B best dev CE | no-mem ms/step | prompt-mem ms/step |",
             "|---|" + "---:|" * 12]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines += ["", f"Generated from {R} and {M}, {len(rows)} tasks."]
    out = ROOT / "results" / "final_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
