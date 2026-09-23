#!/usr/bin/env python3
"""Interactive closed-loop ALFWorld evaluator (official protocol aligned).

Protocol: one episode = one official json_2.1.1 game from the valid_unseen
split. The deterministic PDDL world model (reused from gen_data_alfworld.py)
renders observations and executes commands; the policy picks one command per
step from a gen_data-style candidate set (gold + template distractors,
deterministically sampled per (seed, episode, step)). Mistakes are allowed:
the episode only ends when the goal predicates are satisfied (success) or the
50-step budget is exhausted (fail) -- matching the official ALFWorld
interactive evaluation (max_steps=50, success judged on the final goal only).

Goal predicates follow the alfred goal-PDDL templates, instance-level, with
the involved entities identified from the official walkthrough:
  pick_and_place_simple        (isIn obj recep)
  pick_two_obj_and_place       (isIn obj1 recep) (isIn obj2 recep)
  pick_heat_then_place...      (isIn obj recep) (isHot obj)
  pick_cool_then_place...      (isIn obj recep) (isCool obj)
  pick_clean_then_place...     (isIn obj recep) (isClean obj)
  look_at_obj_in_light         (nextTo obj lamp) (isOn lamp)

Policies (jet_policy.py interface act(obs, candidates) -> key):
  teacher    -- follows the walkthrough (world-model feasibility self-check;
                must be 100% success or the executor/goal check is buggy)
  nomem      -- step-local scoring
  promptmem  -- sliding-window text history
  jet        -- streaming latent memory. Uses a subclass whose act() routes
                leaf scoring through train_jet_bs.score_leaves_bs
                (BroadcastNoStoreCache): the stock JetPolicy.act inline
                DynamicCache view crashes on transformers>=5 when the leaf
                batch > 1 (see the jet06 logs of the game grid).

Usage:
  python eval_alfworld_closed.py --policy teacher --episodes 134 --out t.json
  python eval_alfworld_closed.py --policy nomem --checkpoint ../../checkpoints/NanoJev-unified \
      --model-name base --episodes 10 --out nomem.json --mem-fraction 0.25
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jet.gen_data_alfworld import (
    CAND_HARD_CAP, CAND_TARGET_MAX, CAND_TARGET_MIN, EVENT_MAX, OBS_MAX,
    World, apply_command, candidate_pool, demangle_walkthrough, describe,
    intro_text, summarize_event, truncate_middle,
)

DEFAULT_GAMES = "/data/yangyuming/Long-Jev/data/benchmarks/alfworld/raw/games_valid_unseen.jsonl"
MAX_STEPS = 50  # official ALFWorld interactive protocol budget

# write window each Jet model was trained with (must match at inference)
TRAIN_WINDOW = {"af_jet": 6, "jet17_alfworld": 3, "jet06_mix": 6, "jet17_mix": 2}


# ---------------------------------------------------------------------------
# Goal derivation (alfred goal-PDDL templates, instance level)
# ---------------------------------------------------------------------------
def parse_walkthrough(walkthrough):
    takes, moves, heats, cools, cleans, uses = [], [], [], [], [], []
    for cmd in walkthrough:
        if cmd.startswith("take "):
            m = re.match(r"take (.+) from (.+)$", cmd)
            takes.append(m.group(1))
        elif cmd.startswith("move "):
            m = re.match(r"move (.+) to (.+)$", cmd)
            moves.append((m.group(1), m.group(2)))
        elif cmd.startswith("heat "):
            heats.append(re.match(r"heat (.+) with (.+)$", cmd).group(1))
        elif cmd.startswith("cool "):
            cools.append(re.match(r"cool (.+) with (.+)$", cmd).group(1))
        elif cmd.startswith("clean "):
            cleans.append(re.match(r"clean (.+) with (.+)$", cmd).group(1))
        elif cmd.startswith("use "):
            uses.append(cmd[len("use "):])
    return {"takes": takes, "moves": moves, "heats": heats, "cools": cools,
            "cleans": cleans, "uses": uses}


def derive_goal(rec, walkthrough):
    tt = rec["task_type"]
    p = parse_walkthrough(walkthrough)
    if tt == "look_at_obj_in_light":
        return {"type": "look", "obj": p["takes"][0], "lamp": p["uses"][0]}
    if tt == "pick_and_place_simple":
        objs = [p["moves"][-1][0]]
    elif tt == "pick_two_obj_and_place":
        objs = p["takes"][:2]
    else:
        # processed tasks: the manipulated object is the one finally moved,
        # or (truncated walkthroughs) the one heated/cooled/cleaned
        objs = [p["moves"][-1][0]] if p["moves"] else \
            [(p["heats"] or p["cools"] or p["cleans"])[-1]]
    if p["moves"]:
        recep, recep_type = p["moves"][-1][1], None
    else:
        # 4/134 official walkthroughs are truncated after the process step;
        # the PDDL goal receptacle instance is unrecoverable, so accept any
        # receptacle of the type named in the task description.
        recep, recep_type = None, re.search(r"(?:in|on) ([a-z]+)\.$", rec["task_desc"]).group(1)
    flag = {"pick_heat_then_place_in_recep": "hot",
            "pick_cool_then_place_in_recep": "cold",
            "pick_clean_then_place_in_recep": "clean"}.get(tt)
    return {"type": "place", "objs": objs, "recep": recep, "recep_type": recep_type,
            "flag": flag}


# ---------------------------------------------------------------------------
# Interactive environment on top of the gen_data world model
# ---------------------------------------------------------------------------
class AlfworldEnv:
    def __init__(self, rec):
        self.rec = rec
        self.w = World(rec)
        self.walkthrough = demangle_walkthrough(self.w, rec["walkthrough"])
        self.state = {"at": None, "holding": None}
        self.flags = {}      # obj lower-id -> {"hot","cold","clean"} (persistent)
        self.lamps_on = set()  # lamp display names turned on via `use`
        self.goal = derive_goal(rec, self.walkthrough)
        if self.goal["type"] == "place" and not self.goal.get("recep"):
            # 4/134 official walkthroughs are truncated after the process step;
            # append the missing placement so gold injection / teacher replay
            # work uniformly (any same-type receptacle satisfies the goal).
            r = self.goal_recep_disps()[0]
            obj = self.goal["objs"][0]
            self.walkthrough = self.walkthrough + [f"go to {r}", f"move {obj} to {r}"]
        self.t = 0
        self.done = False
        self.success = False
        self.obs = intro_text(self.w, rec["task_desc"])

    # -- goal predicates ----------------------------------------------------
    def goal_recep_disps(self):
        """Receptacle display names satisfying the goal receptacle: the exact
        instance, or every receptacle of the task_desc-named type."""
        g = self.goal
        if g.get("recep"):
            return [g["recep"]]
        return [d for d in self.w.recep_disps_sorted
                if self.w._name(self.w.by_disp[d]) == g.get("recep_type")]

    def _obj_loc(self, disp):
        """Location id of an object: agent's location if held, else the
        location of the receptacle containing it."""
        if self.state["holding"] == disp:
            return self.w.agent_loc
        oid = self.w.by_disp[disp]
        for o, r in self.w.in_recep:
            if o == oid:
                return self.w.loc_of.get(r)
        return None

    def goal_satisfied(self):
        g = self.goal
        if g["type"] == "place":
            rids = {self.w.by_disp[d] for d in self.goal_recep_disps()}
            for obj in g["objs"]:
                oid = self.w.by_disp[obj]
                if not any(o == oid and r in rids for o, r in self.w.in_recep):
                    return False
                if g["flag"] and g["flag"] not in self.flags.get(oid, set()):
                    return False
            return True
        # look_at_obj_in_light: (nextTo obj lamp) and (isOn lamp)
        if g["lamp"] not in self.lamps_on:
            return False
        lo = self._obj_loc(g["obj"])
        ll = self._obj_loc(g["lamp"])
        return lo is not None and lo == ll

    # -- command legality (same preconditions convert_game enforces) --------
    def executable(self, cmd):
        w, state = self.w, self.state
        try:
            if cmd.startswith("go to "):
                return cmd[len("go to "):] in w.is_recep
            if cmd.startswith("take "):
                m = re.match(r"take (.+) from (.+)$", cmd)
                return (m.group(2) in w.colocated(m.group(2))
                        and m.group(1) in w.contents(m.group(2)))
            if cmd.startswith("move "):
                m = re.match(r"move (.+) to (.+)$", cmd)
                return (state["holding"] == m.group(1)
                        and m.group(2) in w.colocated(m.group(2)))
            if cmd.split()[0] in ("open", "close", "heat", "cool", "clean", "examine"):
                tail = cmd.split(None, 1)[1]
                r = tail.rsplit(" with ", 1)[-1] if " with " in tail else tail
                return r not in w.is_recep or r in w.colocated(r)
            if cmd.startswith("use "):
                return w.obj_at_agent_loc(cmd[len("use "):])
        except Exception:
            return False
        return False

    # -- transition ----------------------------------------------------------
    def step(self, cmd):
        try:
            if cmd.startswith("heat "):
                oid = self.w.by_disp[re.match(r"heat (.+) with (.+)$", cmd).group(1)]
                self.flags.setdefault(oid, set()).add("hot")
            elif cmd.startswith("cool "):
                oid = self.w.by_disp[re.match(r"cool (.+) with (.+)$", cmd).group(1)]
                self.flags.setdefault(oid, set()).add("cold")
            elif cmd.startswith("clean "):
                oid = self.w.by_disp[re.match(r"clean (.+) with (.+)$", cmd).group(1)]
                self.flags.setdefault(oid, set()).add("clean")
            elif cmd.startswith("use "):
                self.lamps_on.add(cmd[len("use "):])
            feedback = apply_command(self.w, self.state, cmd)
        except Exception:
            feedback = "Nothing happens."
        self.t += 1
        if self.goal_satisfied():
            self.done = self.success = True
        elif self.t >= MAX_STEPS:
            self.done = True
        return feedback


# ---------------------------------------------------------------------------
# Candidate construction (gen_data style: gold + template distractors)
# ---------------------------------------------------------------------------
def build_candidates(env, gold, rng, pool=None):
    if pool is None:
        pool = candidate_pool(env.w, env.state)
    k = rng.randint(CAND_TARGET_MIN, CAND_TARGET_MAX)
    if gold is not None:
        rest = [c for c in pool if c != gold]
        cand = [gold] + rng.sample(rest, min(k, len(rest), CAND_HARD_CAP - 1))
    else:
        cand = rng.sample(pool, min(k + 1, len(pool), CAND_HARD_CAP))
    rng.shuffle(cand)
    return {c: describe(c) for c in cand}


class TeacherPolicy:
    """Walkthrough follower; used only for the feasibility self-check."""
    name = "teacher"

    def __init__(self):
        self.gold = None

    def act(self, obs, candidates, feedback=None):
        if self.gold is not None and self.gold in candidates:
            return self.gold
        return next(iter(candidates))


def load_policy(args):
    if args.policy == "teacher":
        return TeacherPolicy()
    from jet_policy import JetPolicy, NoMemPolicy, PromptMemPolicy
    if args.policy == "nomem":
        return NoMemPolicy(args.checkpoint, "alfworld", mem_fraction=args.mem_fraction)
    if args.policy == "promptmem":
        return PromptMemPolicy(args.checkpoint, "alfworld", mem_fraction=args.mem_fraction)
    if args.policy == "jet":
        return FixedJetPolicy(args.checkpoint, "alfworld",
                              note_window=TRAIN_WINDOW.get(args.model_name, 6),
                              note_window_override=TRAIN_WINDOW.get(args.model_name, 6),
                              mem_fraction=args.mem_fraction)
    raise ValueError(args.policy)


class FixedJetPolicy:
    """JetPolicy with act() routed through train_jet_bs.score_leaves_bs.

    Delegate everything else (writer bundle split, streaming memory, remember
    semantics) to the stock JetPolicy; only the batched leaf forward is
    replaced because the inline DynamicCache view in JetPolicy.act crashes on
    transformers>=5 for candidate batches > 1.
    """
    name = "jet"

    def __init__(self, *a, **kw):
        from jet_policy import JetPolicy
        self._inner = JetPolicy(*a, **kw)

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def act(self, obs, candidates, feedback=None):
        from common import leaf_text, tokenize_segments
        from train_jet_bs import score_leaves_bs
        pol = self._inner
        seg = tokenize_segments(pol.tok, [f"State:\n{obs}\n"])
        if pol.ec.length() + len(seg) + 64 > pol.runtime.limit:
            pol.ec.rebuild(0, pol.note_cap)  # emergency compress: keep slots only
        pol.ec.add_segment("step", seg, grad=False)
        keys = list(candidates)
        leaf_ids = [pol.tok.encode(leaf_text(k, candidates[k]), add_special_tokens=False)
                    + [pol.tok.eos_token_id] for k in keys]
        z = score_leaves_bs(pol.model, leaf_ids, pol.ec.cache, pol.device,
                            grad=False, pos_start=pol.ec.pos)
        return keys[int(z.argmax().item())]


def reset_policy(pol, task="alfworld"):
    """Per-episode reset of streaming/prompt memory (see eval_games_closed.py)."""
    if hasattr(pol, "_inner"):  # FixedJetPolicy
        reset_policy(pol._inner, task)
        return
    if hasattr(pol, "ec"):
        from bench_common import TASK_HEADERS
        from common import tokenize_segments
        from train_mem_generic import EpisodeCacheC6
        from unified_game_pipeline import POLICY_QUESTION
        pol.ec = EpisodeCacheC6(pol.model, pol.tok, pol.device, None)
        pol.ec.add_segment("header", tokenize_segments(
            pol.tok, [TASK_HEADERS[task],
                      f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]), grad=False)
        pol.steps_in_window = 0
    if hasattr(pol, "hist"):
        pol.hist = []


def run_episode(rec, pol, seed, ep_idx, trace=None):
    env = AlfworldEnv(rec)
    rng = random.Random(f"{seed}|{ep_idx}")
    ptr = 0  # walkthrough pointer: advances on exact match, so recovery after
              # a detour naturally re-syncs the injected gold
    obs = env.obs
    while not env.done:
        pool = candidate_pool(env.w, env.state)
        gold = None
        if ptr < len(env.walkthrough):
            nxt = env.walkthrough[ptr]
            if nxt in pool or env.executable(nxt):
                gold = nxt
        cands = build_candidates(env, gold, rng, pool)
        if isinstance(pol, TeacherPolicy):
            pol.gold = gold
        obs_in = truncate_middle(obs, OBS_MAX)
        action = pol.act(obs_in, cands)
        feedback = env.step(action)
        event = summarize_event(action, feedback)[:EVENT_MAX]
        if ptr < len(env.walkthrough) and action == env.walkthrough[ptr]:
            ptr += 1
        if hasattr(pol, "remember"):
            pol.remember(obs_in, action, event)
        obs = feedback
        if trace is not None:
            trace.append({"t": env.t, "obs": obs_in, "candidates": cands,
                          "action": action, "event": event, "gold": gold})
    return env.success, env.t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", default=DEFAULT_GAMES)
    p.add_argument("--policy", choices=["teacher", "nomem", "promptmem", "jet"], required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model-name", default=None, help="tag / TRAIN_WINDOW key for jet")
    p.add_argument("--episodes", type=int, default=134)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", required=True)
    p.add_argument("--mem-fraction", type=float, default=0.4)
    p.add_argument("--save-traces", default=None, help="optional dir for per-episode traces")
    a = p.parse_args()
    if a.policy != "teacher" and not a.checkpoint:
        p.error("--checkpoint is required for non-teacher policies")

    games = [json.loads(l) for l in open(a.games, encoding="utf-8") if l.strip()]
    n = min(a.episodes, len(games))
    # deterministic stride sample; identity when n == len(games)
    idxs = [i * len(games) // n for i in range(n)] if n < len(games) else list(range(len(games)))
    sel = [games[i] for i in idxs]

    pol = load_policy(a)
    print(f"policy={a.policy} model={a.model_name or a.checkpoint} episodes={n} "
          f"games={a.games} window={getattr(pol, 'note_window', '-')}", flush=True)
    if a.save_traces:
        Path(a.save_traces).mkdir(parents=True, exist_ok=True)

    succ, steps_all, steps_succ = 0, [], []
    by_type = {}
    t0 = time.perf_counter()
    for i, rec in enumerate(sel):
        trace = [] if a.save_traces else None
        ok, nsteps = run_episode(rec, pol, a.seed, i, trace)
        reset_policy(pol)
        succ += int(ok)
        steps_all.append(nsteps)
        if ok:
            steps_succ.append(nsteps)
        tt = rec["task_type"]
        d = by_type.setdefault(tt, {"n": 0, "succ": 0, "steps": []})
        d["n"] += 1
        d["succ"] += int(ok)
        d["steps"].append(nsteps)
        if trace is not None:
            Path(a.save_traces, f"{i:03d}_{rec['task_type']}.json").write_text(
                json.dumps({"episode_id": rec["id"], "task_type": tt, "success": ok,
                            "n_steps": nsteps, "steps": trace}, ensure_ascii=False, indent=1),
                encoding="utf-8")
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  [{i + 1}/{n}] success {succ}/{i + 1}", flush=True)
    wall = time.perf_counter() - t0

    result = {
        "task": "alfworld", "split": "valid_unseen", "policy": a.policy,
        "model": a.model_name or a.checkpoint, "episodes": n,
        "max_steps": MAX_STEPS,
        "success_rate": succ / n,
        "successes": succ,
        "mean_steps_all": sum(steps_all) / n,
        "mean_steps_success": (sum(steps_succ) / len(steps_succ)) if steps_succ else None,
        "by_task_type": {tt: {"n": d["n"], "success_rate": d["succ"] / d["n"],
                              "mean_steps": sum(d["steps"]) / d["n"]}
                         for tt, d in sorted(by_type.items())},
        "sec_per_episode": wall / n,
    }
    print(json.dumps(result, indent=2), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
