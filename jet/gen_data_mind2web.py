"""Convert Mind2Web task instances into Jeτ episode JSONL.

Episode = one annotation instance (annotation_id) from the official Mind2Web
release (osunlp/Mind2Web). Each action step becomes a choice point: candidates
are 1 positive element + up to 6 negative elements sampled deterministically,
the gold action is the positive's key. Observations stay step-local (website,
task, step counter, previous action_repr); multi-step history is the memory
arm's job.

Input files are top-level JSON arrays (train_N.json / test_domain_N.json) and
are read incrementally with raw_decode so the ~600MB shards never need to fit
in RAM as one text blob.

Split rules (documented in the benchmark README):
  * train/dev: sample from the same pool of official train shards. The pool is
    sorted by annotation_id and shuffled once with random.Random(seed); train
    takes the first N_train episodes, dev the next N_dev. Running this script
    twice with the same --input list and --seed therefore yields disjoint,
    reproducible splits.
  * test: sampled from test_domain/test_domain_0.json, the official
    cross-domain test split.

Usage:
  python gen_data_mind2web.py --input a.json [b.json ...] --split train \
      --episodes 400 --seed 17 --out train.jsonl
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

MAX_STEPS = 30            # instances longer than this are truncated
MAX_NEG = 6               # negatives sampled per step (1 pos + 6 neg = 7 cands)
MIN_CANDS = 2             # hard floor: 1 pos + >=1 neg; below that the step is dropped
MAX_CANDS = 10            # hard ceiling from the unified spec
OBS_LIMIT = 3500
EVENT_LIMIT = 200
TEXT_KEYS = ("aria_label", "title", "placeholder", "input_value", "value",
             "text_value", "alt", "name")   # first non-empty wins as visible text


def stream_instances(path):
    """Yield instances from a top-level JSON array file with bounded memory."""
    dec = json.JSONDecoder()
    buf = ""
    started = False
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            while True:
                stripped = buf.lstrip()
                if not stripped:
                    buf = ""
                    break
                buf = stripped
                if not started:
                    if buf[0] != "[":
                        raise ValueError(f"{path}: expected JSON array, got {buf[:30]!r}")
                    buf, started = buf[1:], True
                    continue
                if buf[0] == ",":
                    buf = buf[1:]
                    continue
                if buf[0] == "]":
                    return
                try:
                    obj, end = dec.raw_decode(buf)
                except ValueError:
                    break  # need more input
                yield obj
                buf = buf[end:]


def _clip(s, n):
    s = " ".join(str(s).split())
    return s[:n].rstrip()


def cand_desc(cand):
    """eval_mind2web cand_desc style for the raw release: tag | id/role | text | attrs."""
    tag = (cand.get("tag") or "?").lower()
    try:
        attrs = json.loads(cand.get("attributes") or "{}")
        if not isinstance(attrs, dict):
            attrs = {}
    except Exception:
        attrs = {}
    bits = [tag]
    ident = attrs.get("id") or attrs.get("name")
    if ident:
        bits.append(f"id={_clip(ident, 40)}")
    role = attrs.get("role") or attrs.get("type")
    if role:
        bits.append(f"role={_clip(role, 24)}")
    for k in TEXT_KEYS:
        txt = attrs.get(k)
        if txt:
            bits.append(f"text: {_clip(txt, 120)}")
            break
    cls = attrs.get("class")
    if cls:
        bits.append(f"class={_clip(cls, 50)}")
    return " | ".join(bits)


def step_event(action_repr, op_value):
    """Derive a short event string from the action_repr '... -> OP[: value]' suffix."""
    if not action_repr:
        return "performed action."
    if " -> " in action_repr:
        elem, tail = action_repr.rsplit(" -> ", 1)
    else:
        elem, tail = action_repr, ""
    op = tail.split(":")[0].strip().upper() or "CLICK"
    elem = _clip(elem, 130)
    if op == "TYPE":
        return _clip(f"typed '{_clip(op_value, 60)}' into {elem}.", EVENT_LIMIT)
    if op == "SELECT":
        return _clip(f"selected '{_clip(op_value, 60)}' in {elem}.", EVENT_LIMIT)
    if op == "HOVER":
        return _clip(f"hovered over {elem}.", EVENT_LIMIT)
    return _clip(f"clicked {elem}.", EVENT_LIMIT)


def build_episode(inst, idx, split, seed):
    """Convert one annotation instance; returns (episode, n_dropped_steps) or (None, reason)."""
    website = inst.get("website", "")
    task = _clip(inst.get("confirmed_task") or inst.get("goal") or "", 400)
    reprs = list(inst.get("action_reprs") or [])
    steps, dropped, prev_repr = [], 0, None
    truncated = False
    for j, act in enumerate(inst.get("actions", [])):
        if len(steps) >= MAX_STEPS:
            truncated = True
            break
        pos_list = act.get("pos_candidates") or []
        negs_all = act.get("neg_candidates") or []
        if not pos_list or not negs_all:
            dropped += 1  # no annotated target or no distractors: cannot form a choice
            continue
        pos = pos_list[0]
        ar = (act.get("action_reprs") or reprs[j:j + 1] or [None])[0]
        ar = str(ar) if ar is not None else ""
        op_value = (act.get("operation") or {}).get("value") or ""
        # deterministic per-step rng: same seed + action_uid -> same sample & order
        rng = random.Random(f"{seed}:{act.get('action_uid', j)}")
        pos_id = pos.get("backend_node_id")
        pos_desc = cand_desc(pos)
        pool, seen_ids, seen_descs = [], {pos_id}, {pos_desc}
        order = list(range(len(negs_all)))
        rng.shuffle(order)  # negative sampling keeps original-file determinism via seed
        for k in order:
            if len(pool) >= MAX_NEG:
                break
            neg = negs_all[k]
            nid, ndesc = neg.get("backend_node_id"), cand_desc(neg)
            if nid in seen_ids or ndesc in seen_descs:
                continue
            seen_ids.add(nid)
            seen_descs.add(ndesc)
            pool.append(ndesc)
        cands = [pos_desc] + pool
        if len(cands) < MIN_CANDS:
            dropped += 1
            continue
        rng.shuffle(cands)  # positive position is uniform, not fixed-first
        keys = [f"option_{i + 1}" for i in range(len(cands))]
        candidates = dict(zip(keys, cands))
        action = keys[cands.index(pos_desc)]
        t = len(steps) + 1
        prev_line = "none (first step)" if prev_repr is None else _clip(prev_repr, 240)
        dom = "/".join(x for x in (inst.get("domain"), inst.get("subdomain")) if x)
        obs = (f"Website: {website}" + (f" ({dom})" if dom else "") + f".\nTask: {task}"
               + f"\nStep {t} of MAX_STEPS_PLACEHOLDER"
               + f"\nPrevious action: {prev_line}"
               + "\nChoose the element to interact with next.")
        steps.append({"t": t, "obs": obs, "candidates": candidates,
                      "action": action, "event": step_event(ar, op_value)})
        prev_repr = ar
    if not steps:
        return None, "no valid steps"
    total = len(steps)
    for s in steps:  # fill in the real episode length
        s["obs"] = s["obs"].replace("MAX_STEPS_PLACEHOLDER", str(total))
    ep = {
        "task": "mind2web",
        "episode_id": f"mind2web-{split}-{idx:04d}",
        "split": split,
        "meta": {
            "website": website,
            "instance_id": inst.get("annotation_id", ""),
            "confirmed_task": task,
            "domain": inst.get("domain", ""),
            "subdomain": inst.get("subdomain", ""),
            "truncated": truncated,
            "dropped_steps": dropped,
        },
        "success": True,
        "n_steps": total,
        "steps": steps,
    }
    return ep, dropped


def validate(ep):
    """Hard schema assertions for one episode."""
    assert ep["task"] == "mind2web" and ep["success"] is True
    assert ep["n_steps"] == len(ep["steps"]) and 1 <= ep["n_steps"] <= MAX_STEPS
    for i, s in enumerate(ep["steps"]):
        assert s["t"] == i + 1, f"bad t in {ep['episode_id']}"
        cands = s["candidates"]
        assert MIN_CANDS <= len(cands) <= MAX_CANDS, f"cand count {len(cands)}"
        assert s["action"] in cands, f"action key missing in {ep['episode_id']} step {s['t']}"
        assert len(list(cands)) == len(set(cands)), "duplicate keys"
        assert len(set(cands.values())) == len(cands), "duplicate candidate descs"
        assert len(s["obs"]) <= OBS_LIMIT, f"obs too long ({len(s['obs'])})"
        assert len(s["event"]) <= EVENT_LIMIT, f"event too long ({len(s['event'])})"
    # positive candidate must be exactly the one the action key points at:
    # enforced by construction (action = keys[index(pos_desc)]) and re-checked here.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True, help="raw Mind2Web json array file(s)")
    p.add_argument("--split", required=True, choices=["train", "dev", "test"])
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--heldout", type=int, default=0,
                   help="with --split train: also materialize this many extra valid "
                        "episodes so the dev run can pick them up (train writes only --episodes)")
    p.add_argument("--skip", type=int, default=0,
                   help="drop the first N valid episodes of the seeded shuffle (used by "
                        "--split dev to take the heldout tail of the train pool)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    # Build the eligible-instance pool deterministically: sorted by annotation_id.
    pool, seen_ids = [], set()
    for path in a.input:
        for inst in stream_instances(path):
            aid = inst.get("annotation_id")
            if not aid or aid in seen_ids:
                continue
            n_act = len(inst.get("actions") or [])
            if n_act == 0:
                continue
            seen_ids.add(aid)
            pool.append(inst)
    pool.sort(key=lambda r: r["annotation_id"])
    rng = random.Random(a.seed)
    rng.shuffle(pool)

    need = a.episodes + (a.heldout if a.split == "train" else a.skip)
    picked, skipped = [], Counter()
    for inst in pool:
        if len(picked) >= need:
            break
        ep, info = build_episode(inst, len(picked), a.split, a.seed)
        if ep is None:
            skipped[info] += 1
            continue
        picked.append(ep)
    if len(picked) < need:
        print(f"WARNING: pool exhausted, only {len(picked)}/{need} episodes", file=sys.stderr)
    picked = picked[a.skip:a.skip + a.episodes]
    for i, ep in enumerate(picked):  # renumber episode ids within the split
        ep["episode_id"] = f"mind2web-{a.split}-{i:04d}"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ep in picked:
            validate(ep)
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # ---- stats ----
    n_ep = len(picked)
    all_steps = [s for ep in picked for s in ep["steps"]]
    step_hist = Counter(min(ep["n_steps"], 30) // 5 * 5 for ep in picked)
    web_top = Counter(ep["meta"]["website"] for ep in picked).most_common(5)
    cand_counts = Counter(len(s["candidates"]) for s in all_steps)
    drop_total = sum(ep["meta"]["dropped_steps"] for ep in picked)
    trunc_total = sum(1 for ep in picked if ep["meta"]["truncated"])
    stats = {
        "file": str(out), "split": a.split, "seed": a.seed, "inputs": a.input,
        "episodes": n_ep,
        "mean_steps": round(sum(ep["n_steps"] for ep in picked) / max(n_ep, 1), 2),
        "mean_candidates": round(sum(len(s["candidates"]) for s in all_steps) / max(len(all_steps), 1), 2),
        "steps_hist(bucket5)": dict(sorted(step_hist.items())),
        "candidates_hist": dict(sorted(cand_counts.items())),
        "websites_top5": web_top,
        "episodes_truncated_at_30": trunc_total,
        "steps_dropped_no_pos_or_neg": drop_total,
        "skipped_instances": dict(skipped),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
