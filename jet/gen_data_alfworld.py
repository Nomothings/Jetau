"""Generate Jeτ episode JSONL for the ALFWorld benchmark.

Source: official ALFWorld json_2.1.1 games (TextWorld + PDDL). Each episode is one
complete expert trajectory replayed against a deterministic world model parsed from
the game's PDDL problem; commands come verbatim from the official game walkthrough,
and observation/feedback strings follow the official alfred.twl2 grammar templates
(json_2.1.1 wording: "move X to Y" for putting, "use X" for toggling lamps).

Episode schema (one JSON object per line):
{"task":"alfworld","episode_id":"alfworld-<split>-<idx>","split":"...",
 "meta":{"game_file":...,"task_type":...},"success":true,"n_steps":N,
 "steps":[{"t":1,"obs":"...","candidates":{"<cmd>":"<desc>",...},"action":"<cmd>","event":"..."}]}

Distractor candidates are instantiated from the world state with ALFWorld command
templates, deterministically sampled per (seed, episode, step).

Usage:
  python gen_data_alfworld.py --raw-dir <dir> --split train --episodes 400 \
      --seed 13 --out alfworld_train.jsonl
Expects <dir>/games_train.jsonl, <dir>/games_valid_seen.jsonl, <dir>/games_valid_unseen.jsonl
(split mapping: train->train, dev->valid_seen, test->valid_unseen).
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

SPLIT_FILE = {"train": "games_train.jsonl", "dev": "games_valid_seen.jsonl",
              "test": "games_valid_unseen.jsonl"}

OBS_MAX = 3500   # chars, truncate middle keeping head+tail
EVENT_MAX = 200  # chars
CAND_HARD_CAP = 10
CAND_TARGET_MIN, CAND_TARGET_MAX = 4, 7  # distractors added on top of gold


# --------------------------------------------------------------------------
# World model (demangling replicates alfworld/agents/utils/misc.py Demangler)
# --------------------------------------------------------------------------
class World:
    def __init__(self, rec):
        def ok(i):
            return re.match(r"^[A-Za-z][A-Za-z0-9]*", i) and "?" not in i

        self.receptacles = {i: t for i, t, _o in rec["receptacles"] if ok(i)}
        self.objects = {i: t for i, t in rec["objects"] if ok(i)}
        self.openable = {i.lower() for i, _t, o in rec["receptacles"] if o and ok(i)}
        self.opened = {i.lower() for i in rec.get("opened0", [])}
        # NOTE: the official PDDL init can assert the same object in several
        # receptacles (duplicate inReceptacle facts, ~2% of facts); the TextWorld
        # engine treats them as a fact set and would render the object in each
        # receptacle, so we keep a list of pairs instead of a dict.
        self.in_recep = []  # [obj_lower_id, recep_lower_id]
        for o, r in rec["in_recep"]:
            if o in self.objects and r in self.receptacles:
                self.in_recep.append((o.lower(), r.lower()))
        # numbering: lowercase ids, sort, assign per name pool via pop() from the end
        ids = sorted([i.lower() for i in self.receptacles] + [i.lower() for i in self.objects])
        counts = Counter()
        for i in ids:
            counts[self._name(i)] += 1
        pool = {n: list(range(c + 1))[1:] for n, c in counts.items()}
        self.disp = {}
        for i in ids:
            self.disp[i] = "{} {}".format(self._name(i), pool[self._name(i)].pop())
        self.by_disp = {v: k for k, v in self.disp.items()}
        self.recep_disps = [self.disp[i.lower()] for i in self.receptacles]
        self.recep_disps_sorted = sorted(self.recep_disps)
        self.is_recep = {d: True for d in self.recep_disps}
        # location model: PDDL preconditions (PickupObject/PutObject/...) only require
        # the agent to share a location with the target receptacle; several
        # receptacles can share one location.
        self.loc_of = {}
        for r, l in rec.get("recep_loc", []):
            if r in self.receptacles:
                self.loc_of[r.lower()] = l
        self.at_loc_receps = {}
        for r in self.receptacles:
            self.at_loc_receps.setdefault(self.loc_of.get(r.lower()), []).append(self.disp[r.lower()])
        self.agent_loc = None

    def colocated(self, recep_disp):
        """Receptacles sharing the agent's current location (empty if agent not colocated)."""
        if self.agent_loc is None:
            return []
        return self.at_loc_receps.get(self.agent_loc, [])

    @staticmethod
    def _name(lower_id):
        name = lower_id.split("_bar_", 1)[0]
        if "basin" in lower_id:
            name += "basin"
        return name

    def contents(self, recep_disp):
        """Object display names directly in receptacle (visible iff open or not openable)."""
        rid = self.by_disp[recep_disp]
        if rid in self.openable and rid not in self.opened:
            return []
        return self.all_contents(recep_disp)

    def all_contents(self, recep_disp):
        rid = self.by_disp[recep_disp]
        return sorted({self.disp[o] for o, r in self.in_recep if r == rid})

    def remove_obj(self, obj_disp):
        oid = self.by_disp[obj_disp]
        self.in_recep = [(o, r) for o, r in self.in_recep if o != oid]

    def obj_at_agent_loc(self, obj_disp):
        """True if the object sits in a receptacle at the agent's location."""
        if self.agent_loc is None:
            return False
        oid = self.by_disp[obj_disp]
        return any(self.loc_of.get(r) == self.agent_loc
                   for o, r in self.in_recep if o == oid)

    def place_obj(self, obj_disp, recep_disp):
        self.remove_obj(obj_disp)
        self.in_recep.append((self.by_disp[obj_disp], self.by_disp[recep_disp]))


# --------------------------------------------------------------------------
# Text rendering (official alfred.twl2 grammar templates, json_2.1.1 wording)
# --------------------------------------------------------------------------
def join_and(items):
    items = list(items)
    if not items:
        return "nothing"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]


def examine_text(w, r):
    rid = w.by_disp[r]
    if rid in w.openable:
        if rid in w.opened:
            objs = [f"a {o}" for o in w.all_contents(r)]
            return f"The {r} is open. In it, you see {join_and(objs)}."
        return f"The {r} is closed."
    objs = [f"a {o}" for o in w.all_contents(r)]
    return f"On the {r}, you see {join_and(objs)}."


def intro_text(w, task_desc):
    receps = [f"a {r}" for r in w.recep_disps_sorted]
    return ("-= Welcome to TextWorld, ALFRED! =-\n\n"
            "You are in the middle of a room. Looking quickly around you, you see "
            f"{join_and(receps)}.\n\nYour task is to: {task_desc}")


# command -> feedback (also mutates world state)
def apply_command(w, state, cmd):
    """state: {'at': recep_display|None, 'holding': obj_display|None}. Returns feedback."""
    toks = cmd.split()
    if cmd.startswith("go to "):
        r = cmd[len("go to "):]
        state["at"] = r
        w.agent_loc = w.loc_of.get(w.by_disp[r])
        return f"You arrive at {r}. {examine_text(w, r)}"
    if cmd.startswith("take "):
        m = re.match(r"take (.+) from (.+)$", cmd)
        o, r = m.group(1), m.group(2)
        w.remove_obj(o)
        state["holding"] = o
        return f"You pick up the {o} from the {r}."
    if cmd.startswith("move "):
        m = re.match(r"move (.+) to (.+)$", cmd)
        o, r = m.group(1), m.group(2)
        w.place_obj(o, r)
        state["holding"] = None
        return f"You move the {o} to the {r}."
    if cmd.startswith("open "):
        r = cmd[len("open "):]
        rid = w.by_disp[r]
        w.opened.add(rid)
        return f"You open the {r}. {examine_text(w, r)}"
    if cmd.startswith("close "):
        r = cmd[len("close "):]
        w.opened.discard(w.by_disp[r])
        return f"You close the {r}."
    if cmd.startswith("use "):
        o = cmd[len("use "):]
        return f"You turn on the {o}."
    if cmd.startswith("heat "):
        m = re.match(r"heat (.+) with (.+)$", cmd)
        return f"You heat the {m.group(1)} using the {m.group(2)}."
    if cmd.startswith("cool "):
        m = re.match(r"cool (.+) with (.+)$", cmd)
        return f"You cool the {m.group(1)} using the {m.group(2)}."
    if cmd.startswith("clean "):
        m = re.match(r"clean (.+) with (.+)$", cmd)
        return f"You clean the {m.group(1)} using the {m.group(2)}."
    if cmd.startswith("examine "):
        x = cmd[len("examine "):]
        if x in w.is_recep:
            return examine_text(w, x)
        return f"There's nothing special about the {x}."
    raise ValueError(f"unsupported command: {cmd}")


OBJ_LIST_RE = re.compile(r"a (\S+ \d+)(?:,| and|\.)")


def summarize_event(cmd, feedback):
    """Short (<=200 chars) plain description of the action result."""
    def n_obj(items):
        return f"{len(items)} object{'s' if len(items) != 1 else ''}"

    if cmd.startswith("go to "):
        r = cmd[6:]
        if f"The {r} is closed." in feedback:
            det = "it is closed"
        elif f"The {r} is open." in feedback:
            det = "it is open with {} inside".format(n_obj(OBJ_LIST_RE.findall(feedback)))
        else:
            det = "on it you see {}".format(n_obj(OBJ_LIST_RE.findall(feedback)))
        return f"moved to the {r}; {det}."
    if cmd.startswith("take "):
        m = re.match(r"take (.+) from (.+)$", cmd)
        return f"picked up the {m.group(1)} from the {m.group(2)}."
    if cmd.startswith("move "):
        m = re.match(r"move (.+) to (.+)$", cmd)
        return f"moved the {m.group(1)} to the {m.group(2)}."
    if cmd.startswith("open "):
        r = cmd[5:]
        n = len(re.findall(r"a \S+ \d+", feedback.split("In it, you see", 1)[-1])) if "In it" in feedback else 0
        return f"opened the {r}; {n} object{'s' if n != 1 else ''} inside."
    if cmd.startswith("close "):
        return f"closed the {cmd[6:]}."
    if cmd.startswith("use "):
        return f"turned on the {cmd[4:]}."
    if cmd.startswith("heat "):
        m = re.match(r"heat (.+) with (.+)$", cmd)
        return f"heated the {m.group(1)} with the {m.group(2)}."
    if cmd.startswith("cool "):
        m = re.match(r"cool (.+) with (.+)$", cmd)
        return f"cooled the {m.group(1)} with the {m.group(2)}."
    if cmd.startswith("clean "):
        m = re.match(r"clean (.+) with (.+)$", cmd)
        return f"cleaned the {m.group(1)} with the {m.group(2)}."
    if cmd.startswith("examine "):
        return f"examined the {cmd[8:]}."
    return feedback


DESC = [
    (re.compile(r"^go to (.+)$"), lambda m: f"go to the {m.group(1)}"),
    (re.compile(r"^take (.+) from (.+)$"), lambda m: f"take the {m.group(1)} from the {m.group(2)}"),
    (re.compile(r"^move (.+) to (.+)$"), lambda m: f"move the {m.group(1)} to the {m.group(2)}"),
    (re.compile(r"^open (.+)$"), lambda m: f"open the {m.group(1)}"),
    (re.compile(r"^close (.+)$"), lambda m: f"close the {m.group(1)}"),
    (re.compile(r"^use (.+)$"), lambda m: f"turn on the {m.group(1)}"),
    (re.compile(r"^heat (.+) with (.+)$"), lambda m: f"heat the {m.group(1)} with the {m.group(2)}"),
    (re.compile(r"^cool (.+) with (.+)$"), lambda m: f"cool the {m.group(1)} with the {m.group(2)}"),
    (re.compile(r"^clean (.+) with (.+)$"), lambda m: f"clean the {m.group(1)} with the {m.group(2)}"),
    (re.compile(r"^examine (.+)$"), lambda m: f"examine the {m.group(1)}"),
]


def describe(cmd):
    for pat, fn in DESC:
        m = pat.match(cmd)
        if m:
            return fn(m)
    return cmd


# --------------------------------------------------------------------------
# Candidate (distractor) construction
# --------------------------------------------------------------------------
def candidate_pool(w, state):
    """Syntactically valid, engine-executable ALFWorld commands given the context:
    interaction commands target receptacles that share the agent's location
    (PDDL precondition), navigation targets every other receptacle."""
    pool = []
    holding = state["holding"]
    here = w.colocated(state["at"]) if state["at"] is not None else []
    for r in w.recep_disps_sorted:
        if r not in here:
            pool.append(f"go to {r}")
    for at in sorted(here):
        rid = w.by_disp[at]
        visible = w.contents(at)
        if holding is None:
            for o in visible:
                if "lamp" not in o:
                    pool.append(f"take {o} from {at}")
        else:
            pool.append(f"move {holding} to {at}")
            if at.startswith("microwave"):
                pool.append(f"heat {holding} with {at}")
            elif at.startswith("fridge"):
                pool.append(f"cool {holding} with {at}")
            elif at.endswith("basin"):
                pool.append(f"clean {holding} with {at}")
        if rid in w.openable:
            pool.append(f"open {at}" if rid not in w.opened else f"close {at}")
        pool.append(f"examine {at}")
        if "desklamp" in at or "floorlamp" in at:
            pool.append(f"use {at}")
    if holding is not None:
        pool.append(f"examine {holding}")
    return pool


def truncate_middle(s, limit):
    if len(s) <= limit:
        return s
    head, tail = int(limit * 0.6), int(limit * 0.35)
    return s[:head] + " ...[truncated]... " + s[-tail:]


# --------------------------------------------------------------------------
# Episode conversion
# --------------------------------------------------------------------------
def demangle_walkthrough(w, walkthrough):
    """Some games ship walkthrough commands with raw PDDL entity ids; normalize
    them to demangled display names ('desk_bar__...' -> 'desk 2')."""
    out = []
    for cmd in walkthrough:
        toks = [w.disp.get(t, t) if "_bar_" in t else t for t in cmd.split()]
        out.append(" ".join(toks))
    return out


def convert_game(rec, idx, split, seed):
    w = World(rec)
    walkthrough = demangle_walkthrough(w, rec["walkthrough"])
    state = {"at": None, "holding": None}
    obs = intro_text(w, rec["task_desc"])
    steps = []
    for t, cmd in enumerate(walkthrough, 1):
        # expert precondition checks (PDDL domain): agent must share a location
        # with the target receptacle; object must be visible in it, etc.
        if cmd.startswith("go to "):
            r = cmd[len("go to "):]
            if r not in w.is_recep:
                raise ValueError(f"goto unknown receptacle: {cmd}")
        elif cmd.startswith("take "):
            m = re.match(r"take (.+) from (.+)$", cmd)
            if m.group(2) not in w.colocated(m.group(2)) or m.group(1) not in w.contents(m.group(2)):
                raise ValueError(f"take precondition failed: {cmd}")
        elif cmd.startswith("move "):
            m = re.match(r"move (.+) to (.+)$", cmd)
            if state["holding"] != m.group(1) or m.group(2) not in w.colocated(m.group(2)):
                raise ValueError(f"move precondition failed: {cmd}")
        elif cmd.split()[0] in ("open", "close", "heat", "cool", "clean", "examine"):
            tail = cmd.split(None, 1)[1]
            r = tail.rsplit(" with ", 1)[-1] if " with " in tail else tail
            if r in w.is_recep and r not in w.colocated(r):
                raise ValueError(f"precondition failed (not colocated): {cmd}")
        elif cmd.startswith("use "):
            o = cmd[len("use "):]
            if not w.obj_at_agent_loc(o):
                raise ValueError(f"use precondition failed: {cmd}")
        pool = [c for c in candidate_pool(w, state) if c != cmd]
        rng = random.Random(f"{seed}|{split}|{idx}|{t}")
        k = rng.randint(CAND_TARGET_MIN, CAND_TARGET_MAX)
        cand_cmds = [cmd]
        if pool:
            take = min(k, len(pool), CAND_HARD_CAP - 1)
            cand_cmds += rng.sample(pool, take)
        rng.shuffle(cand_cmds)
        candidates = {c: describe(c) for c in cand_cmds}
        feedback = apply_command(w, state, cmd)
        steps.append({
            "t": t,
            "obs": truncate_middle(obs, OBS_MAX),
            "candidates": candidates,
            "action": cmd,
            "event": summarize_event(cmd, feedback)[:EVENT_MAX],
        })
        obs = feedback
    return {
        "task": "alfworld",
        "episode_id": f"alfworld-{split}-{idx}",
        "split": split,
        "meta": {"game_file": rec["game_file_path"], "task_type": rec["task_type"]},
        "success": True,
        "n_steps": len(steps),
        "steps": steps,
    }


def validate(ep):
    assert ep["task"] == "alfworld" and ep["success"] is True
    assert re.match(r"^alfworld-(train|dev|test)-\d+$", ep["episode_id"])
    assert ep["split"] in ("train", "dev", "test")
    assert isinstance(ep["meta"]["game_file"], str) and ep["meta"]["task_type"]
    assert ep["n_steps"] == len(ep["steps"]) and ep["n_steps"] >= 1
    for i, st in enumerate(ep["steps"], 1):
        assert st["t"] == i
        assert isinstance(st["obs"], str) and 0 < len(st["obs"]) <= OBS_MAX
        assert 3 <= len(st["candidates"]) <= CAND_HARD_CAP, \
            f"{ep['episode_id']} t={i}: {len(st['candidates'])} candidates"
        assert st["action"] in st["candidates"]
        assert 0 < len(st["event"]) <= EVENT_MAX
        for k, v in st["candidates"].items():
            assert isinstance(k, str) and isinstance(v, str) and k and v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    games = []
    with open(raw / SPLIT_FILE[args.split], encoding="utf-8") as f:
        for line in f:
            games.append(json.loads(line))
    games.sort(key=lambda g: (g["task_type"], g["game_file_path"]))
    rng = random.Random(args.seed)
    rng.shuffle(games)
    n = min(args.episodes, len(games))

    eps, dropped = [], 0
    for g in games:
        if len(eps) >= n:
            break
        try:
            ep = convert_game(g, len(eps), args.split, args.seed)
            validate(ep)
            eps.append(ep)
        except Exception as e:  # unparsable/inconsistent game: skip honestly
            dropped += 1
            if dropped <= 5:
                print(f"DROP {g['game_file_path']}: {e}")

    Path(args.out).write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps), encoding="utf-8")

    # ---- statistics -----------------------------------------------------
    steps = [s for e in eps for s in e["steps"]]
    n_steps = [e["n_steps"] for e in eps]
    ncand = [len(s["candidates"]) for s in steps]
    obs_lens = [len(s["obs"]) for s in steps]
    dist = Counter()
    for x in n_steps:
        dist["1-3" if x <= 3 else "4-6" if x <= 6 else "7-9" if x <= 9 else "10+"] += 1
    print(f"[{args.split}] episodes={len(eps)} (requested {args.episodes}, pool {len(games)}, dropped {dropped})")
    print(f"  steps: mean={sum(n_steps)/len(n_steps):.2f} min={min(n_steps)} max={max(n_steps)} dist={dict(sorted(dist.items()))}")
    print(f"  candidates/step: mean={sum(ncand)/len(ncand):.2f} min={min(ncand)} max={max(ncand)}")
    print(f"  obs chars: mean={sum(obs_lens)/len(obs_lens):.0f} max={max(obs_lens)}; event chars: mean={sum(len(s['event']) for s in steps)/len(steps):.0f}")
    print(f"  task types: {dict(Counter(e['meta']['task_type'] for e in eps))}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
