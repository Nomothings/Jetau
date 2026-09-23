#!/usr/bin/env python3
"""Evaluate interactive ALFWorld tasks with a deterministic PDDL world model."""
import argparse
import json
import random
import re
import time
from pathlib import Path

from jet.gen_data_alfworld import (
    CAND_HARD_CAP, CAND_TARGET_MAX, CAND_TARGET_MIN, EVENT_MAX, OBS_MAX,
    World, apply_command, candidate_pool, demangle_walkthrough, describe,
    intro_text, summarize_event, truncate_middle,
)

DEFAULT_GAMES = "data/alfworld/raw/games_valid_unseen.jsonl"
MAX_STEPS = 50  # official ALFWorld interactive protocol budget

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
    from jet.jet_policy import JetPolicy, NoMemPolicy, PromptMemPolicy
    if args.policy == "nomem":
        return NoMemPolicy(args.checkpoint, "alfworld", mem_fraction=args.mem_fraction)
    if args.policy == "promptmem":
        return PromptMemPolicy(args.checkpoint, "alfworld", mem_fraction=args.mem_fraction)
    if args.policy == "jet":
        return JetPolicy(args.checkpoint, "alfworld",
                         note_window=args.note_window,
                         mem_fraction=args.mem_fraction)
    raise ValueError(args.policy)


def reset_policy(pol, task="alfworld"):
    if hasattr(pol, "reset"):
        pol.reset()


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
    p.add_argument("--model-name", default=None, help="label for the output record")
    p.add_argument("--note-window", type=int, default=6)
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
