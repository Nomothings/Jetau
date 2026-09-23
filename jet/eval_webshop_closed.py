#!/usr/bin/env python3
"""Interactive closed-loop WebShop evaluator (official protocol semantics).

The agent starts on the WebShop search page and interacts with the offline
WebShop replica environment, reusing every mechanism verbatim from
gen_data_webshop.py (BM25 retrieval over the ~50k-product catalog, search /
results / item page observations, candidate action construction, and the
exact official reward formula ported from web_agent_site/engine/goal.py):

    search[<query>]  ->  click[item[<asin>]]  ->
    click[option[<name>: <value>]]  ->  buy[now]

An episode ends on `buy[now]` (reward = official formula on the purchased
item with the currently selected options) or at the action cap
(--max-actions, default 10). No purchase within the cap => reward 0.

Policies:
  teacher    rule shopper ported 1:1 from gen_data_webshop.run_episode
             (eps=0). Self-check: reward ~0.98 on the first 50 official test
             queries (offline generator stats: 0.9758 over the full 500).
  nomem      step-local scoring (released NanoJev)
  promptmem  sliding-window text history in the prompt
  jet        streaming latent memory with the trained write head

Metrics: reward_mean, reward_p50, success_rate (strict: reward >= 0.999),
mean_actions, buy_rate (+ reward_min and the soft 0.7-threshold rate).
"""
import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jet.gen_data_webshop import (
    DISPLAY_RESULTS, EVENT_LIMIT, BM25, TOKEN_RE, build_goals, get_reward,
    instr_keywords, load_corpus, md5_seed, name_keywords, obs_item_page,
    obs_results_page, obs_search_page, token_set_ratio, trunc,
)

MAX_ACTIONS = 10
SUCCESS_STRICT = 0.999     # r == 1.0 full match
SUCCESS_SOFT = 0.7         # task-card dataset threshold, reported for reference

# write window each Jet model was trained with (must match at inference;
# from each model's train_log.json config)
TRAIN_WINDOW = {"ws_jet": 6, "jet17_webshop": 3, "jet06_mix": 6, "jet17_mix": 2,
                "jet06_maze": 6, "jet17_maze": 3, "jet06_snake": 6,
                "jet17_snake": 3, "jet06_pokemon": 6, "jet17_pokemon": 2}

DEFAULT_DATA_DIR = "/data/yangyuming/Long-Jev/data/benchmarks/webshop/raw"
DEFAULT_CACHE = "/data/yangyuming/tmp/webshop_closed_corpus.pkl"


# ------------------------------------------------------------------ corpus
def load_world(data_dir, cache_path, num_products=50000):
    """products, goals, bm25, asins, query_pool.

    The first build scans the full items_shuffle.json (~10 min, ~40 GB RAM);
    the result is cached to a pickle so subsequent runs start in seconds.
    BM25 is cached with the corpus (tf counters are the expensive part).
    """
    if cache_path and Path(cache_path).exists():
        t0 = time.time()
        with open(cache_path, "rb") as f:
            world = pickle.load(f)
        print(f"[corpus] loaded cache {cache_path} ({time.time()-t0:.0f}s)",
              flush=True)
    else:
        products, goal_asins, human_ins = load_corpus(data_dir, num_products)
        goals = build_goals(products, goal_asins, human_ins)
        for i, g in enumerate(goals):
            g["query_id"] = i
        asins = list(products.keys())
        print(f"[bm25] indexing {len(asins)} products ...", flush=True)
        t0 = time.time()
        docs = []
        for a in asins:
            p = products[a]
            option_text = ', and '.join(
                f"{k}: {', '.join(v)}" for k, v in p['options'].items())
            contents = ' '.join([p['Title'], p['Description'],
                                 p['BulletPoints'][0] if p['BulletPoints'] else '',
                                 option_text]).lower()
            docs.append(TOKEN_RE.findall(contents))
        bm25 = BM25(docs)
        del docs
        print(f"[bm25] done ({time.time()-t0:.0f}s)", flush=True)
        world = {"products": products, "goals": goals, "bm25": bm25}
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            tmp = str(cache_path) + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(world, f, protocol=4)
            Path(tmp).rename(cache_path)
            print(f"[corpus] wrote cache {cache_path}", flush=True)
    products, goals, bm25 = world["products"], world["goals"], world["bm25"]
    asins = list(products.keys())
    query_pool = [products[a]['query'] for a in asins[:20000]
                  if products[a]['query']]
    return products, goals, bm25, asins, query_pool


# ------------------------------------------------------- interactive session
class WebShopSession:
    """One interactive WebShop episode.

    Mechanics (state machine, observations, candidate construction, event
    strings) are ported 1:1 from gen_data_webshop.run_episode so that the
    interactive trajectory distribution matches the offline training data.
    Candidate keys are built together with an intent side map so actions
    never need string re-parsing:
      key -> ('search', q) | ('item', asin) | ('option', (name, value))
           | ('buy', None).
    """

    def __init__(self, goal, products, bm25, asins, query_pool, split, idx,
                 max_actions=MAX_ACTIONS):
        self.goal, self.products, self.bm25 = goal, products, bm25
        self.asins, self.query_pool = asins, query_pool
        self.split, self.idx = split, idx
        self.max_actions = max_actions
        # same episode rng stream as the offline generator (eps = 0)
        self.rng = random.Random(md5_seed('ep', split, goal['query_id'], idx, 0))
        self.instr = goal['instruction_text']
        self.kw = instr_keywords(self.instr)
        self.refine_q = name_keywords(goal['name'])          # 8-token title keywords
        self.refine_q2 = name_keywords(goal['name'][:200])   # wider variant
        self.distractors = self.rng.sample(query_pool, 4)

        self.page = 'search'            # 'search' | 'results' | 'item'
        self.selected_options = {}      # option name -> value
        self.clicked_asin = None
        self.searches_done = 0
        self.used_queries = set()
        self.query, self.shown, self.n_hits = None, [], 0
        self.fallback_mode = False      # goal unreachable: commit & buy
        self.t = 0                      # actions taken
        self.bought = False
        self.done = False
        self.reward = 0.0
        self._version = 0
        self._cand_version = -1
        self._cand, self._intent = {}, {}

    # -- observations (identical to the offline replica pages) --------------
    def obs(self):
        if self.page == 'search':
            return obs_search_page(self.instr)
        if self.page == 'results':
            return obs_results_page(self.instr, self.query, self.shown,
                                    self.n_hits)
        return obs_item_page(self.instr, self.products[self.clicked_asin])

    # -- candidates (identical construction to the offline replica) ---------
    def candidates(self):
        if self._cand_version != self._version:
            self._build_candidates()
        return dict(self._cand)

    def _build_candidates(self):
        cand, intent = {}, {}
        if self.page == 'search':
            cand_queries = [self.kw, self.goal['query']] + self.distractors[:3]
            ordered, seen_q = [], set()
            for q in cand_queries:
                if q and q not in seen_q:
                    seen_q.add(q)
                    ordered.append(q)
            for q in ordered:
                key = f"search[{q}]"
                cand[key] = f"search for '{trunc(q, 60)}'"
                intent[key] = ('search', q)

        elif self.page == 'results':
            for a in [x for x, _, _ in self.shown][:7]:
                key = f"click[item[{a}]]"
                cand[key] = (f"open item: {trunc(self.products[a]['Title'], 60)} "
                             f"({self.products[a]['Price']})")
                intent[key] = ('item', a)
            alt_searches, tried = [], 0
            for q in [self.kw, self.refine_q, self.refine_q2,
                      self.goal['query']] + self.distractors:
                if len(alt_searches) >= 1:
                    break
                if q and q not in self.used_queries:
                    alt_searches.append(q)
            while not alt_searches and tried < 50:
                tried += 1
                q = self.rng.choice(self.query_pool)
                if q not in self.used_queries:
                    alt_searches.append(q)
            for q in alt_searches:
                key = f"search[{q}]"
                cand[key] = f"search for '{trunc(q, 60)}' instead"
                intent[key] = ('search', q)

        else:  # item page
            prod = self.products[self.clicked_asin]
            pending = self._pending_option()
            if pending:
                values = prod['options'][pending]
                goal_vals = [v for v in values
                             if any(token_set_ratio(v, g_opt) > 85
                                    for g_opt in self.goal['goal_options'])][:6]
                others = [v for v in values if v not in goal_vals]
                if len(values) > 6:
                    rng_o = random.Random(md5_seed('opts', self.clicked_asin,
                                                   pending))
                    k = max(0, min(6 - len(goal_vals), len(others)))
                    keep = set(goal_vals) | set(rng_o.sample(others, k))
                    show_vals = [v for v in values if v in keep]
                else:
                    show_vals = values
                for v in show_vals:
                    key = f"click[option[{pending}: {v}]]"
                    cand[key] = f"choose {pending} '{trunc(v, 40)}'"
                    intent[key] = ('option', (pending, v))
                cand["buy[now]"] = "buy this item with current options"
                intent["buy[now]"] = ('buy', None)
                back = f"search[{self.refine_q}]"
                cand[back] = "go back and search again"
                intent[back] = ('search', self.refine_q)
            else:
                # buy decision page
                cand["buy[now]"] = f"buy this item now ({prod['Price']})"
                intent["buy[now]"] = ('buy', None)
                alts = [a for a, _, _ in self.shown if a != self.clicked_asin][:1]
                if not alts:
                    alts = [a for a in self.asins[:80]
                            if a != self.clicked_asin][:1]
                for a in alts:
                    key = f"click[item[{a}]]"
                    cand[key] = (f"open item: "
                                 f"{trunc(self.products[a]['Title'], 50)} instead")
                    intent[key] = ('item', a)
                back_q = next((q for q in [self.refine_q, self.refine_q2,
                                           self.goal['query']] + self.distractors
                               if q and q not in self.used_queries),
                              self.goal['query'])
                back = f"search[{back_q}]"
                cand[back] = "search again for the requested product"
                intent[back] = ('search', back_q)
        self._cand, self._intent = cand, intent
        self._cand_version = self._version

    def _pending_option(self):
        """First option name with an unselected goal value (offline replica)."""
        prod = self.products[self.clicked_asin]
        for g_opt in self.goal['goal_options']:
            if any(token_set_ratio(v, g_opt) > 85
                   for v in self.selected_options.values()):
                continue
            for name, values in prod['options'].items():
                if name in self.selected_options:
                    continue
                if any(token_set_ratio(v, g_opt) > 85 for v in values):
                    return name
        return None

    # -- environment transition ----------------------------------------------
    def _do_search(self, q):
        self.query = q
        self.used_queries.add(q)
        self.searches_done += 1
        hits = self.bm25.search(q, self.asins)
        self.n_hits = len(hits)
        self.shown = [(a, self.products[a]['Title'], self.products[a]['Price'])
                      for a in hits[:DISPLAY_RESULTS]]
        return trunc(f"searched for '{q}'; {self.n_hits} results.", EVENT_LIMIT)

    def step(self, key):
        cand = self.candidates()          # ensures built exactly once/state
        assert key in cand, (key, self.page)
        kind, payload = self._intent[key]
        if kind == 'search':
            event = self._do_search(payload)
            self.page = 'results'
        elif kind == 'item':
            prev = self.page
            self.clicked_asin = payload
            self.page = 'item'
            if prev == 'item':
                event = f"switched to item {payload}."
            else:
                event = (f"opened item {payload}: "
                         f"{trunc(self.products[payload]['Title'], 80)}.")
        elif kind == 'option':
            name, value = payload
            self.selected_options[name] = value
            event = f"chose option '{name}: {value}'."
        else:  # buy
            self.bought = True
            self.done = True
            prod = self.products[self.clicked_asin]
            self.reward = get_reward(prod, self.goal, prod['price'],
                                     self.selected_options)
            event = (f"bought item {self.clicked_asin} "
                     f"({trunc(prod['Title'], 60)}); reward {self.reward:.2f}.")
        self._version += 1
        self.t += 1
        return trunc(event, EVENT_LIMIT)

    # -- teacher (rule shopper, eps=0; port of run_episode's rule branches) --
    def _best_options_for(self, prod):
        opts = {}
        for g_opt in self.goal['goal_options']:
            for name, values in prod['options'].items():
                if name in opts:
                    continue
                hit = next((v for v in values
                            if token_set_ratio(v, g_opt) > 85), None)
                if hit is not None:
                    opts[name] = hit
                    break
        return opts

    def _reward_if_bought(self, asin):
        prod = self.products[asin]
        return get_reward(prod, self.goal, prod['price'],
                          self._best_options_for(prod))

    def teacher_action(self):
        cand = self.candidates()
        intent = self._intent
        keys = list(cand)
        if self.page == 'search':
            rule = (f"search[{self.kw}]" if self.kw
                    else f"search[{self.goal['query']}]")
            if rule not in cand:
                rule = keys[0]
        elif self.page == 'results':
            goal_key = f"click[item[{self.goal['asin']}]]"
            refine = ([self.refine_q, self.refine_q2]
                      if self.refine_q not in self.used_queries
                      else [self.refine_q2])
            if goal_key in cand:
                rule = goal_key
            elif self.searches_done < 3 and any(f"search[{q}]" in cand
                                                for q in refine):
                rule = next(f"search[{q}]" for q in refine
                            if f"search[{q}]" in cand)
            else:
                item_keys = [k for k in keys if intent[k][0] == 'item']
                if item_keys:
                    rule = max(item_keys,
                               key=lambda k: self._reward_if_bought(
                                   intent[k][1]))
                else:                      # empty results page
                    rule = next((k for k in keys if intent[k][0] == 'search'),
                                keys[0])
                self.fallback_mode = True
        else:  # item page
            prod = self.products[self.clicked_asin]
            pending = self._pending_option()
            if pending:
                values = prod['options'][pending]
                goal_vals = [v for v in values
                             if any(token_set_ratio(v, g_opt) > 85
                                    for g_opt in
                                    self.goal['goal_options'])][:6]
                rule = (f"click[option[{pending}: {goal_vals[0]}]]"
                        if goal_vals else "buy[now]")
                if rule not in cand:
                    rule = "buy[now]"
            else:
                is_goal = self.clicked_asin == self.goal['asin']
                back_q = next((q for q in [self.refine_q, self.refine_q2,
                                           self.goal['query']] + self.distractors
                               if q and q not in self.used_queries),
                              self.goal['query'])
                rule = ("buy[now]" if is_goal or self.fallback_mode
                        or self.searches_done >= 3 else f"search[{back_q}]")
                if rule not in cand:
                    rule = "buy[now]"
        # eps=0 explore_or: consumes exactly one rng draw per action, keeping
        # this interactive trajectory byte-identical to the offline generator
        self.rng.random()
        return rule


# ---------------------------------------------------------------- policies
def build_policy(args):
    from jet_policy import JetPolicy, NoMemPolicy, PromptMemPolicy
    if args.policy == "nomem":
        return NoMemPolicy(args.checkpoint, "webshop",
                           mem_fraction=args.mem_fraction)
    if args.policy == "promptmem":
        return PromptMemPolicy(args.checkpoint, "webshop",
                               mem_fraction=args.mem_fraction)
    nw = TRAIN_WINDOW.get(args.model_name, 6)
    return JetPolicy(args.checkpoint, "webshop", note_window=nw,
                     note_window_override=nw, mem_fraction=args.mem_fraction)


def reset_policy(pol):
    """Reset per-episode memory state (streaming cache / text history)."""
    if hasattr(pol, "ec"):
        from bench_common import TASK_HEADERS
        from common import tokenize_segments
        from train_mem_generic import EpisodeCacheC6
        from unified_game_pipeline import POLICY_QUESTION
        pol.ec = EpisodeCacheC6(pol.model, pol.tok, pol.device, None)
        pol.ec.add_segment("header", tokenize_segments(
            pol.tok, [TASK_HEADERS["webshop"],
                      f"Question type: choice\nQuestion:\n{POLICY_QUESTION}\n"]),
            grad=False)
        pol.steps_in_window = 0
    if hasattr(pol, "hist"):
        pol.hist = []


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",
                    choices=["teacher", "nomem", "promptmem", "jet"],
                    required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--model-name", default="", help="key into TRAIN_WINDOW for jet")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0,
                    help="start index within the official bucket")
    ap.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    ap.add_argument("--mem-fraction", type=float, default=0.4)
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--corpus-cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    products, goals, bm25, asins, query_pool = load_world(
        args.data_dir, args.corpus_cache)

    # official split buckets (random.Random(233) shuffle already applied)
    buckets = {"test": goals[0:500], "dev": goals[500:1500]}
    bucket = buckets[args.split]
    assert args.offset + args.episodes <= len(bucket), \
        f"{args.split} bucket has {len(bucket)} goals"

    pol = None
    if args.policy != "teacher":
        pol = build_policy(args)
        print(f"policy={args.policy} model={args.model_name or args.checkpoint} "
              f"task=webshop window={getattr(pol, 'note_window', '-')}",
              flush=True)
    else:
        print("policy=teacher (rule shopper, eps=0) task=webshop", flush=True)

    rows = []
    t0 = time.perf_counter()
    for i in range(args.offset, args.offset + args.episodes):
        goal = bucket[i]
        env = WebShopSession(goal, products, bm25, asins, query_pool,
                             args.split, i, args.max_actions)
        if args.policy == "teacher":
            while not env.done and env.t < env.max_actions:
                env.step(env.teacher_action())
        else:
            obs = env.obs()
            while not env.done and env.t < env.max_actions:
                cands = env.candidates()
                action = pol.act(obs, cands)
                if action not in cands:      # defensive: policy must pick a key
                    action = list(cands)[0]
                event = env.step(action)
                if hasattr(pol, "remember"):
                    pol.remember(obs, action, event)
                obs = env.obs()
            reset_policy(pol)
        reward = env.reward if env.bought else 0.0
        rows.append({
            "episode_id": f"webshop-{args.split}-{i:04d}",
            "query_id": goal["query_id"],
            "reward": round(reward, 4),
            "actions": env.t,
            "bought": env.bought,
            "bought_asin": env.clicked_asin if env.bought else None,
            "success": bool(reward >= SUCCESS_STRICT),
            "success_soft": bool(reward >= SUCCESS_SOFT),
        })
        n_done = i - args.offset + 1
        if n_done % 25 == 0 or n_done == args.episodes:
            done_r = sum(r["reward"] for r in rows)
            print(f"[{n_done}/{args.episodes}] reward_mean={done_r/n_done:.4f} "
                  f"buy_rate={sum(r['bought'] for r in rows)/n_done:.3f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    wall = time.perf_counter() - t0
    n = max(len(rows), 1)
    rewards = sorted(r["reward"] for r in rows)
    result = {
        "task": "webshop",
        "protocol": ("interactive closed-loop; official reward on buy; "
                     f"reward=0 if not bought within {args.max_actions} actions"),
        "policy": args.policy,
        "model": args.model_name or args.checkpoint or "teacher",
        "split": args.split,
        "episodes": len(rows),
        "max_actions": args.max_actions,
        "reward_mean": round(sum(rewards) / n, 4),
        "reward_p50": rewards[n // 2],
        "reward_min": rewards[0],
        "success_rate": round(sum(r["success"] for r in rows) / n, 4),
        "success_rate_soft07": round(sum(r["success_soft"] for r in rows) / n, 4),
        "mean_actions": round(sum(r["actions"] for r in rows) / n, 3),
        "buy_rate": round(sum(r["bought"] for r in rows) / n, 4),
        "sec_per_episode": round(wall / n, 3),
        "per_episode": rows,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_episode"},
                     indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[write] {args.out}", flush=True)


if __name__ == "__main__":
    main()
