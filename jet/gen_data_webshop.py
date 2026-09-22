#!/usr/bin/env python
"""Offline WebShop episode generator for the Long-Jev benchmark suite.

Produces unified episode JSONL (one episode per line):
  {"task":"webshop","episode_id":"webshop-<split>-<idx>","split":...,
   "meta":{"query_id":...,"instruction":...},"success":bool,"n_steps":N,
   "steps":[{"t":1,"obs":...,"candidates":{key:desc,...},"action":key,"event":...}]}

What this replicates offline (no server, no Java/Lucene, CPU-only):
  * Product corpus: official items_shuffle.json (full 1.18M-product file is
    scanned once to recover the official goal ordering); a ~50k product subset
    (all human-instruction products + file-order fillers, file is pre-shuffled)
    forms the searchable catalog.
  * Goal/split construction: faithful port of
    web_agent_site/engine/goal.py::get_human_goals + SimServer
    (web_agent_site/envs/web_agent_text_env.py): goals built in product-file
    order, then random.Random(233).shuffle(goals), then split by
    baseline_models/env.py: test = goals[0:500], dev = goals[500:1500],
    train = goals[1500:].  Deviations (documented in README):
      - price_upper / ranged prices sampled with per-item md5 seeds (official
        uses the unseeded global random at that point, i.e. nondeterministic).
      - r_type's title_score uses a token-overlap heuristic instead of spaCy
        noun extraction (spaCy unavailable offline); query_match/category_match
        are exact.  Only affects exploration purchases, never teacher buys of
        the goal product (query_match=True => r_type=1).
      - fuzzy attribute/option matching uses difflib.SequenceMatcher, which is
        thefuzz's own fallback implementation path of token_set_ratio.
  * Reward: exact port of web_agent_site/engine/goal.py::get_reward:
        total = (num_attr_matches + num_option_matches + r_price)
                / (len(goal_attrs) + len(goal_options) + 1) * r_type
  * Search: BM25 (Lucene parameters k1=1.2(+ we use 1.5), b=0.75) over the
    same fields official Lucene indexes (Title + Description + BulletPoints[0]
    + option text, lowercased alphanumeric tokens), top-50 like
    SEARCH_RETURN_N=50.
  * Teacher: rule shopper that maximises the official reward
    (search[goal query] -> refine if needed -> click goal item -> select goal
    options -> buy[now]), with small epsilon exploration mixed into train.

Run on A6000:
  python gen_data_webshop.py --data-dir <dir with items_shuffle.json,
      items_ins_v2.json, items_human_ins.json> --out-dir <dir for jsonl>
"""
import argparse
import difflib
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- constants
SEARCH_RETURN_N = 50      # official engine.py
DISPLAY_RESULTS = 10      # official PRODUCT_WINDOW
MAX_STEPS = 8
OBS_LIMIT = 3500
EVENT_LIMIT = 200
SUCCESS_THRESHOLD = 0.7   # task-card definition, see README
PRICE_RANGE = [10.0 * i for i in range(1, 100)]   # official goal.py

STOPWORDS = {
    "i", "im", "am", "is", "are", "looking", "for", "need", "want", "would",
    "like", "a", "an", "the", "with", "and", "of", "to", "in", "on", "please",
    "find", "me", "my", "some", "that", "which", "it", "its", "this",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def md5_seed(*parts):
    return int(hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def trunc(s, n):
    """Collapse horizontal whitespace only; keep line structure; cap length."""
    s = "\n".join(re.sub(r"[ \t]+", " ", ln).strip()
                  for ln in str(s).split("\n"))
    s = re.sub(r"\n{2,}", "\n", s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ------------------------------------------------- thefuzz fallback port
def ratio(a, b):
    """SequenceMatcher ratio — thefuzz's fallback similarity."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def token_set_ratio(a, b):
    """Port of thefuzz.fuzz.token_set_ratio (difflib fallback path)."""
    ta, tb = set(TOKEN_RE.findall(a.lower())), set(TOKEN_RE.findall(b.lower()))
    if not ta and not tb:
        return 100.0
    inter = ta & tb
    s = max(
        ratio(" ".join(sorted(inter | (ta - tb))),
              " ".join(sorted(inter | (tb - ta)))),
        ratio(" ".join(sorted(inter | (ta - tb))), " ".join(sorted(inter))),
        ratio(" ".join(sorted(inter)), " ".join(sorted(inter | (tb - ta)))),
    )
    return 100.0 * s


# ---------------------------------------------------- official normalize.py
COLOR_SET = [
    'alabaster', 'apricot', 'aqua', 'ash', 'asphalt', 'azure', 'banana',
    'beige', 'black', 'blue', 'blush', 'bordeaux', 'bronze', 'brown',
    'burgundy', 'camel', 'camo', 'caramel', 'champagne', 'charcoal',
    'cheetah', 'chestnut', 'chocolate', 'christmas', 'coffee', 'cognac',
    'copper', 'coral', 'cranberry', 'cream', 'crystal', 'dark', 'denim',
    'eggplant', 'elephant', 'espresso', 'fuchsia', 'gold', 'granite',
    'grape', 'graphite', 'grass', 'gray', 'green', 'grey', 'heather',
    'indigo', 'ivory', 'ivy', 'khaki', 'lavender', 'lemon', 'leopard',
    'light', 'lilac', 'lime', 'magenta', 'maroon', 'mauve', 'merlot',
    'midnight', 'mint', 'mocha', 'multicolor', 'mushroom', 'mustard',
    'natural', 'navy', 'nude', 'olive', 'orange', 'peach', 'pewter', 'pink',
    'plum', 'purple', 'rainbow', 'red', 'rose', 'royal', 'rust', 'sand',
    'sapphire', 'seashell', 'silver', 'skull', 'slate', 'steel', 'stone',
    'stonewash', 'sunflower', 'tan', 'taupe', 'teal', 'tiger', 'turquoise',
    'violet', 'walnut', 'wheat', 'white', 'wine', 'yellow',
]


def normalize_color(color_string):
    """Official web_agent_site/engine/normalize.py::normalize_color."""
    for norm_color in COLOR_SET:
        if norm_color in color_string:
            return norm_color
    return color_string


# ------------------------------------------------------- reward (official)
def get_type_reward(purchased_product, goal):
    """Port of goal.py::get_type_reward (title_score: token heuristic)."""
    query_match = purchased_product['query'] == goal['query']

    purchased_cat = [x.strip() for x in purchased_product['product_category'].split('›')]
    goal_cat = [x.strip() for x in goal['product_category'].split('›')]
    category_match = len(set(purchased_cat) & set(goal_cat)) >= 2

    # official: spaCy noun tokens; offline: lowercase alpha tokens minus stopwords
    desired = [t for t in TOKEN_RE.findall(goal['name'].lower()) if t not in STOPWORDS]
    purchased = [t for t in TOKEN_RE.findall(purchased_product['name'].lower())
                 if t not in STOPWORDS]
    n_intersect = len(set(desired) & set(purchased))
    title_score = 0.2 if len(desired) == 0 else n_intersect / len(desired)

    r_type = 1.0
    if not (query_match or category_match or title_score > 0.2):
        r_type = 0.5
    if title_score < 0.1:
        r_type = 0.1
    if title_score == 0.0:
        r_type = 0.0
    return r_type, query_match, category_match, title_score


def get_attribute_reward(purchased_product, goal):
    """Port of goal.py::get_attribute_reward."""
    num_attr_matches = 0
    for g_attr in goal['attributes']:
        matched = False
        for p_attr in purchased_product['Attributes']:
            if token_set_ratio(p_attr, g_attr) > 85:
                num_attr_matches += 1
                matched = True
                break
        if (not matched and (
                g_attr in purchased_product['Title'].lower()
                or g_attr in ' '.join(purchased_product['BulletPoints']).lower()
                or g_attr in purchased_product['Description'].lower())):
            num_attr_matches += 1
    return num_attr_matches


def get_option_reward(purchased_options, goal_options):
    """Port of goal.py::get_option_reward."""
    purchased_options = [normalize_color(o) for o in purchased_options]
    goal_iter = (goal_options.items() if isinstance(goal_options, dict)
                 else goal_options)
    goal_opts = [normalize_color(g if isinstance(g, str) else g[1])
                 for g in goal_iter]
    num_option_matches = 0
    for g_option in goal_opts:
        for p_option in purchased_options:
            if token_set_ratio(p_option, g_option) > 85:
                num_option_matches += 1
                break
    return num_option_matches, len(goal_opts)


def get_reward(purchased_product, goal, price, options):
    """Exact port of goal.py::get_reward (non-verbose branch)."""
    r_type, _, _, _ = get_type_reward(purchased_product, goal)
    r_price = 1 if price <= goal['price_upper'] else 0
    num_attr_matches = get_attribute_reward(purchased_product, goal)
    goal_opts = (goal['goal_options'].items() if isinstance(goal['goal_options'], dict)
                 else goal['goal_options'])
    n_goal_opts = len(list(goal_opts))
    num_option_matches, _ = get_option_reward(list(options.values()),
                                              goal['goal_options'])
    total = ((num_attr_matches + num_option_matches + r_price)
             / (len(goal['attributes']) + n_goal_opts + 1))
    return total * r_type


# ------------------------------------------------------------ BM25-lite
class BM25:
    """BM25 over lowercased alphanumeric tokens, Lucene-style idf."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf, self.len = [], []
        df = Counter()
        for tokens in docs:
            counts = Counter(tokens)
            self.tf.append(counts)
            self.len.append(len(tokens))
            df.update(counts.keys())
        self.n = len(docs)
        self.avgdl = sum(self.len) / max(1, self.n)
        self.idf = {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5))
                    for t, d in df.items()}

    def search(self, query, asins, k=SEARCH_RETURN_N):
        q = TOKEN_RE.findall(query.lower())
        scores = []
        for i in range(self.n):
            tf = self.tf[i]
            s, dl = 0.0, self.len[i]
            for t in set(q):
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scores.append((s, i))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [asins[i] for _, i in scores[:k]]


# --------------------------------------------------------- data loading
def normalize_product(p, attributes):
    """Port of engine.py::load_products single-item normalization."""
    asin = p['asin']
    rec = {
        'asin': asin,
        'name': p.get('name', ''),
        'Title': p.get('name', ''),
        'Description': p.get('full_description') or '',
        'BulletPoints': (p['small_description']
                         if isinstance(p.get('small_description'), list)
                         else [p.get('small_description') or '']),
        'category': p.get('category', ''),
        'product_category': p.get('product_category', ''),
        'query': (p.get('query') or '').lower().strip(),
    }
    pricing = p.get('pricing')
    if not pricing:
        pricing_floats, price_tag = [100.0], '$100.0'
    else:
        pricing_floats = []
        for price in pricing.split('$')[1:]:
            cleaned = re.sub(r'[^\d.]', '', price)
            if cleaned:
                pricing_floats.append(float(cleaned))
        if not pricing_floats:
            pricing_floats, price_tag = [100.0], '$100.0'
        elif len(pricing_floats) == 1:
            price_tag = f"${pricing_floats[0]}"
        else:
            pricing_floats = pricing_floats[:2]
            price_tag = f"${pricing_floats[0]} to ${pricing_floats[1]}"
    rec['pricing'] = pricing_floats
    rec['Price'] = price_tag
    # deterministic per-asin price inside the range (official: random.uniform)
    if len(pricing_floats) == 1:
        rec['price'] = pricing_floats[0]
    else:
        rec['price'] = random.Random(md5_seed('price', asin)).uniform(*pricing_floats[:2])

    options = {}
    customization_options = p.get('customization_options')
    if customization_options:
        for option_name, option_contents in customization_options.items():
            if option_contents is None:
                continue
            option_name = option_name.lower()
            values = [oc['value'].strip().replace('/', ' | ').lower()
                      for oc in option_contents]
            options[option_name] = values
    rec['options'] = options

    attrs = attributes.get(asin, {}).get('attributes')
    rec['Attributes'] = attrs if attrs else ['DUMMY_ATTR']
    return rec


def load_corpus(data_dir, num_products=50000):
    """Scan the full items_shuffle.json once (official goal order needs it)."""
    data_dir = Path(data_dir)
    t0 = time.time()
    print(f"[load] human instructions ...", flush=True)
    with open(data_dir / 'items_human_ins.json') as f:
        human_ins = json.load(f)
    if isinstance(human_ins, list):          # some mirrors store pairs
        human_ins = dict(human_ins)
    print(f"[load] {len(human_ins)} asins with human instructions "
          f"({time.time()-t0:.0f}s)", flush=True)

    with open(data_dir / 'items_ins_v2.json') as f:
        attributes = json.load(f)

    print(f"[load] streaming items_shuffle.json (full file, for official "
          f"goal order) ...", flush=True)
    t0 = time.time()
    with open(data_dir / 'items_shuffle.json') as f:
        raw_products = json.load(f)
    print(f"[load] {len(raw_products)} raw products ({time.time()-t0:.0f}s)",
          flush=True)

    products, goal_asins_in_order, seen = {}, [], set()
    for p in raw_products:
        asin = p['asin']
        if asin == 'nan' or len(asin) > 10 or asin in seen:
            continue
        seen.add(asin)
        if asin in human_ins:
            rec = normalize_product(p, attributes)
            products[asin] = rec
            goal_asins_in_order.append(asin)
        elif len(products) < num_products:
            products[asin] = normalize_product(p, attributes)
    del raw_products, attributes, seen
    print(f"[load] subset: {len(products)} products "
          f"({len(goal_asins_in_order)} with human goals) "
          f"({time.time()-t0:.0f}s)", flush=True)
    return products, goal_asins_in_order, human_ins


def build_goals(products, goal_asins_in_order, human_ins):
    """Port of goal.py::get_human_goals + SimServer seeding/shuffle."""
    goals = []
    skipped = 0
    for asin in goal_asins_in_order:
        product = products[asin]
        for ins in human_ins[asin]:
            attributes = ins['instruction_attributes']
            if len(attributes) == 0:
                skipped += 1
                continue
            price = product['price']
            price_range = [p for p in PRICE_RANGE if p > price][:4]
            rng = random.Random(md5_seed('price_upper', asin, ins['instruction']))
            if len(price_range) >= 2:
                _, price_upper = sorted(rng.sample(price_range, 2))
                price_text = f', and price lower than {price_upper:.2f} dollars'
            else:
                price_upper, price_text = 1000000, ''
            goals.append({
                'asin': asin,
                'category': product['category'],
                'query': product['query'],
                'name': product['Title'],
                'product_category': product['product_category'],
                'instruction_text': ins['instruction'].strip('.') + price_text,
                'attributes': attributes,
                'price_upper': price_upper,
                'goal_options': list(ins.get('instruction_options') or []),
            })
    # official SimServer: random.seed(233); random.shuffle(goals)
    random.Random(233).shuffle(goals)
    print(f"[goals] {len(goals)} goals ({skipped} skipped, no attributes); "
          f"official split buckets: test=goals[0:500], dev=goals[500:1500], "
          f"train=goals[1500:]", flush=True)
    return goals


# ------------------------------------------------------------ obs pages
def obs_search_page(instruction):
    return (f"Instruction: {instruction}\n\n"
            "WebShop search page. Type a search query to find the product "
            "you need, then buy the right item with the right options.")


def obs_results_page(instruction, query, shown, total):
    lines = [f"Instruction: {instruction}", "",
             f"Search results for '{query}' "
             f"(showing top {len(shown)} of {total} matches):"]
    for i, (asin, title, price) in enumerate(shown, 1):
        lines.append(f"{i}. {asin} | {trunc(title, 110)} | {price}")
    if not shown:
        lines.append("(no results)")
    return trunc("\n".join(lines), OBS_LIMIT)


def obs_item_page(instruction, product):
    p = product
    lines = [f"Instruction: {instruction}", "",
             f"Item: {trunc(p['Title'], 160)}",
             f"Price: {p['Price']}",
             f"Category: {trunc(p['product_category'], 120)}",
             f"Attributes: {trunc('; '.join(p['Attributes']), 200)}"]
    if p['options']:
        lines.append("Options:")
        for name, values in p['options'].items():
            lines.append(f"- {name}: {trunc(', '.join(values), 160)}")
    if p['Description']:
        lines.append(f"Description: {trunc(p['Description'], 700)}")
    return trunc("\n".join(lines), OBS_LIMIT)


def instr_keywords(instruction):
    body = re.sub(r',? and price lower than [\d.]+ dollars', '', instruction)
    toks = [t for t in TOKEN_RE.findall(body.lower()) if t not in STOPWORDS]
    return " ".join(toks[:10])


def name_keywords(title, max_tokens=8):
    toks = [t for t in TOKEN_RE.findall(title.lower()) if t not in STOPWORDS]
    return " ".join(toks[:max_tokens])


# --------------------------------------------------------- teacher rollouts
def run_episode(goal, products, bm25, asins, split, idx, eps, query_pool):
    """Rule shopper: search goal query -> (refine) -> open goal item ->
    select goal options -> buy. Small eps exploration. Returns (episode, reward).

    Candidate keys are built together with an `intent` side map so actions
    never need string re-parsing: key -> ('search', q) | ('item', asin)
    | ('option', (name, value)) | ('buy', None).
    """
    rng = random.Random(md5_seed('ep', split, goal['query_id'], idx,
                                 int(eps * 100)))
    instr = goal['instruction_text']
    kw = instr_keywords(instr)
    refine_q = name_keywords(goal['name'])          # 8-token title keywords
    refine_q2 = name_keywords(goal['name'][:200])   # wider (12-token) variant
    distractors = rng.sample(query_pool, 4)

    steps = []
    page = 'search'                # 'search' | 'results' | 'item'
    selected_options = {}          # option name -> value
    clicked_asin, searches_done, used_queries = None, 0, set()
    query, shown, n_hits, bought = None, [], 0, False
    fallback_mode = False          # goal unreachable: maximise reward & buy

    def best_options_for(prod):
        """Options a reward-maximising shopper would pick on `prod`."""
        opts = {}
        for g_opt in goal['goal_options']:
            for name, values in prod['options'].items():
                if name in opts:
                    continue
                hit = next((v for v in values
                            if token_set_ratio(v, g_opt) > 85), None)
                if hit is not None:
                    opts[name] = hit
                    break
        return opts

    def reward_if_bought(asin):
        prod = products[asin]
        return get_reward(prod, goal, prod['price'], best_options_for(prod))

    def do_search(q):
        nonlocal query, shown, n_hits, searches_done, used_queries
        query = q
        used_queries.add(q)
        searches_done += 1
        hits = bm25.search(q, asins)
        n_hits = len(hits)
        shown = [(a, products[a]['Title'], products[a]['Price'])
                 for a in hits[:DISPLAY_RESULTS]]
        return trunc(f"searched for '{q}'; {n_hits} results.", EVENT_LIMIT)

    def add_step(obs, cand, intent, action, event):
        assert action in cand and action in intent
        steps.append({"t": len(steps) + 1, "obs": trunc(obs, OBS_LIMIT),
                      "candidates": cand, "action": action,
                      "event": trunc(event, EVENT_LIMIT)})

    def explore_or(cand_keys, rule_action):
        """Epsilon-greedy: with prob eps pick a uniformly random candidate."""
        if rng.random() < eps and len(cand_keys) > 1:
            alt = rng.choice(cand_keys)
            if alt != rule_action:
                return alt
        return rule_action

    while not bought and len(steps) < MAX_STEPS - 1:
        if page == 'search':
            cand_queries = [kw, goal['query']] + distractors[:3]
            ordered, seen_q = [], set()
            for q in cand_queries:
                if q and q not in seen_q:
                    seen_q.add(q)
                    ordered.append(q)
            cand, intent = {}, {}
            for q in ordered:
                key = f"search[{q}]"
                cand[key] = f"search for '{trunc(q, 60)}'"
                intent[key] = ('search', q)
            keys = list(cand)
            rule = (f"search[{kw}]" if kw else f"search[{goal['query']}]")
            action = explore_or(keys, rule)
            if action not in cand:
                action = keys[0]
            event = do_search(intent[action][1])
            add_step(obs_search_page(instr), cand, intent, action, event)
            page = 'results'

        elif page == 'results':
            cand, intent = {}, {}
            for a in [x for x, _, _ in shown][:7]:
                key = f"click[item[{a}]]"
                cand[key] = (f"open item: {trunc(products[a]['Title'], 60)} "
                             f"({products[a]['Price']})")
                intent[key] = ('item', a)
            alt_searches, tried = [], 0
            for q in [kw, refine_q, refine_q2, goal['query']] + distractors:
                if len(alt_searches) >= 1:
                    break
                if q and q not in used_queries:
                    alt_searches.append(q)
            while not alt_searches and tried < 50:
                tried += 1
                q = rng.choice(query_pool)
                if q not in used_queries:
                    alt_searches.append(q)
            for q in alt_searches:
                key = f"search[{q}]"
                cand[key] = f"search for '{trunc(q, 60)}' instead"
                intent[key] = ('search', q)
            keys = list(cand)
            goal_key = f"click[item[{goal['asin']}]]"
            if goal_key in cand:
                rule = goal_key
            elif searches_done < 3 and any(
                    f"search[{q}]" in cand
                    for q in ([refine_q, refine_q2] if refine_q not in used_queries
                              else [refine_q2])):
                rule = next(f"search[{q}]" for q in
                            ([refine_q, refine_q2]
                             if refine_q not in used_queries else [refine_q2])
                            if f"search[{q}]" in cand)
            else:
                # goal unreachable on this page: buy the best-reward item
                rule = max((k for k in keys if intent[k][0] == 'item'),
                           key=lambda k: reward_if_bought(intent[k][1]))
                fallback_mode = True
            action = explore_or(keys, rule)
            if action not in cand:
                action = keys[0]
            kind, payload = intent[action]
            if kind == 'search':
                event = do_search(payload)
                add_step(obs_results_page(instr, query, shown, n_hits),
                         cand, intent, action, event)
            else:
                if action != rule:
                    fallback_mode = False   # exploration click, may recover
                clicked_asin = payload
                event = (f"opened item {clicked_asin}: "
                         f"{trunc(products[clicked_asin]['Title'], 80)}.")
                add_step(obs_results_page(instr, query, shown, n_hits),
                         cand, intent, action, event)
                page = 'item'

        elif page == 'item':
            prod = products[clicked_asin]
            is_goal = clicked_asin == goal['asin']
            pending = None     # first option name with an unselected goal value
            for g_opt in goal['goal_options']:
                if any(token_set_ratio(v, g_opt) > 85
                       for v in selected_options.values()):
                    continue
                for name, values in prod['options'].items():
                    if name in selected_options:
                        continue
                    if any(token_set_ratio(v, g_opt) > 85 for v in values):
                        pending = name
                        break
                if pending:
                    break

            cand, intent = {}, {}
            if pending:
                values = prod['options'][pending]
                goal_vals = [v for v in values
                             if any(token_set_ratio(v, g_opt) > 85
                                    for g_opt in goal['goal_options'])][:6]
                others = [v for v in values if v not in goal_vals]
                if len(values) > 6:
                    rng_o = random.Random(md5_seed('opts', clicked_asin, pending))
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
                back = f"search[{refine_q}]"
                cand[back] = "go back and search again"
                intent[back] = ('search', refine_q)
                keys = list(cand)
                rule = (f"click[option[{pending}: {goal_vals[0]}]]"
                        if goal_vals else "buy[now]")
                action = explore_or(keys, rule)
                if action not in cand:
                    action = keys[0]
                kind, payload = intent[action]
                if kind == 'option':
                    selected_options[payload[0]] = payload[1]
                    event = f"chose option '{payload[0]}: {payload[1]}'."
                elif kind == 'buy':
                    bought = True
                    event = "pending"
                else:
                    event = do_search(payload)
                    page = 'results'
                add_step(obs_item_page(instr, prod), cand, intent, action,
                         event)
            else:
                # buy decision page
                cand["buy[now]"] = f"buy this item now ({prod['Price']})"
                intent["buy[now]"] = ('buy', None)
                alts = [a for a, _, _ in shown if a != clicked_asin][:1]
                if not alts:
                    alts = [a for a in asins[:80] if a != clicked_asin][:1]
                for a in alts:
                    key = f"click[item[{a}]]"
                    cand[key] = f"open item: {trunc(products[a]['Title'], 50)} instead"
                    intent[key] = ('item', a)
                back_q = next((q for q in [refine_q, refine_q2, goal['query']]
                               + distractors
                               if q and q not in used_queries), goal['query'])
                back = f"search[{back_q}]"
                cand[back] = "search again for the requested product"
                intent[back] = ('search', back_q)
                keys = list(cand)
                # buy the goal item; from an exploration mis-click recover
                # once; in fallback mode (goal unreachable) commit and buy
                rule = ("buy[now]" if is_goal or fallback_mode
                        or searches_done >= 3 else back)
                action = explore_or(keys, rule)
                if action not in cand:
                    action = keys[0]
                kind, payload = intent[action]
                if kind == 'buy':
                    bought = True
                    event = "pending"
                elif kind == 'item':
                    clicked_asin = payload
                    event = f"switched to item {clicked_asin}."
                else:
                    event = do_search(payload)
                    page = 'results'
                add_step(obs_item_page(instr, prod), cand, intent, action,
                         event)

    if not bought:
        # step budget exhausted: force-buy the currently open item so every
        # episode terminates with a well-defined official reward
        prod = products[clicked_asin]
        cand = {"buy[now]": f"buy this item now ({prod['Price']})"}
        add_step(obs_item_page(instr, prod), cand,
                 {"buy[now]": ('buy', None)}, "buy[now]",
                 "step budget exhausted; bought current item.")

    reward = get_reward(products[clicked_asin], goal,
                        products[clicked_asin]['price'], selected_options)
    for s in reversed(steps):
        if s["event"] == "pending":
            s["event"] = trunc(
                f"bought item {clicked_asin} "
                f"({trunc(products[clicked_asin]['Title'], 60)}); "
                f"reward {reward:.2f}.", EVENT_LIMIT)
            break
    return {
        "task": "webshop",
        "episode_id": f"webshop-{split}-{idx:04d}",
        "split": split,
        "meta": {"query_id": goal['query_id'], "instruction": instr},
        "success": bool(reward >= SUCCESS_THRESHOLD),
        "n_steps": len(steps),
        "steps": steps,
    }, reward


# ---------------------------------------------------------------- self-test
def self_test():
    """Official tests/web-agent-site/engine/test_goal.py cases (spaCy-free)."""
    # attribute reward
    goal = {'attributes': ["tea tree", "essential oils", "natural ingredients"]}
    p = {'Attributes': ["tea tree", "essential oil", "natural ingredients"],
         'Title': "", 'BulletPoints': [], 'Description': ""}
    assert get_attribute_reward(p, goal) == 3, get_attribute_reward(p, goal)
    p = {'Attributes': ["essential oil", "natural ingredients"],
         'Title': "", 'BulletPoints': [], 'Description': ""}
    assert get_attribute_reward(p, goal) == 2
    p = {'Attributes': [], 'Title': "",
         'BulletPoints': ["This shampoo has essential oils and smells like lemons"],
         'Description': "Best shampoo on the market, made with natural ingredients"}
    assert get_attribute_reward(p, goal) == 2
    p = {'Attributes': ["tea bag", "earl gray", "lipton"],
         'Title': "English tea for breakfast",
         'BulletPoints': ["Soothing aroma", "Calming, great feeling"],
         'Description': "Best tea made by Lipton, great to pair with breakfast"}
    assert get_attribute_reward(p, goal) == 0
    # option reward
    g, pr = ["grey", "XL", "pack of 12"], ["pack of 12", "grey", "XL"]
    m, n = get_option_reward(pr, g)
    assert (m, n) == (3, 3)
    m, _ = get_option_reward(["pack of 12", "blue", "XL"], g)
    assert m == 2
    g = ["cool powder snow", "XL", "pack of 12"]
    m, _ = get_option_reward(["pack of 12", "powder snow", "XL"], g)
    assert m == 3, m          # fuzzy match case from official tests
    m, n = get_option_reward(["goal 1", "goal 2"], [])
    assert (m, n) == (0, 0)
    # full reward, exact match case
    goal = {'query': "Query 1", 'product_category': "a › b › c",
            'name': "Mens D.O.N. Issue 2 Gca Basketball Sneakers Shoes Casual - Off White",
            'attributes': ["tea tree", "essential oils", "natural ingredients"],
            'goal_options': {"color": "grey", "size": "XL"}, 'price_upper': 40.00}
    purchased = {'query': "Query 1", 'product_category': "a › b › c",
                 'name': "Mens D.O.N. Issue 2 Gca Basketball Sneakers Shoes Casual - Off White",
                 'Attributes': ["tea tree", "essential oil", "natural ingredients"],
                 'Title': "", 'BulletPoints': [], 'Description': ""}
    r = get_reward(purchased, goal, 35, {"color": "grey", "size": "XL"})
    assert abs(r - 1.0) < 1e-9, r
    # query/category matching exactness
    rt, qm, cm, _ = get_type_reward(
        {'query': "Query 1", 'product_category': "b › c › a", 'name': "X"},
        {'query': "Query 1", 'product_category': "a › b › c", 'name': "Y"})
    assert qm and cm
    print("[self-test] official reward unit cases passed")


# ------------------------------------------------------------------- main
def validate(eps):
    assert eps["task"] == "webshop" and eps["split"] in ("train", "dev", "test")
    assert eps["episode_id"].startswith(f"webshop-{eps['split']}-")
    assert isinstance(eps["meta"]["query_id"], int)
    assert isinstance(eps["meta"]["instruction"], str) and eps["meta"]["instruction"]
    assert isinstance(eps["success"], bool)
    assert eps["n_steps"] == len(eps["steps"]) and eps["n_steps"] >= 1
    for s in eps["steps"]:
        assert set(s) >= {"t", "obs", "candidates", "action", "event"}
        assert s["action"] in s["candidates"], (eps["episode_id"], s["t"])
        assert 1 <= len(s["candidates"]) <= 10
        assert len(s["obs"]) <= OBS_LIMIT, (eps["episode_id"], len(s["obs"]))
        assert len(s["event"]) <= EVENT_LIMIT
        assert len(set(s["candidates"])) == len(s["candidates"])
    json.dumps(eps, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--num-products', type=int, default=50000)
    ap.add_argument('--n-train', type=int, default=400)
    ap.add_argument('--n-dev', type=int, default=60)
    ap.add_argument('--n-test', type=int, default=100)
    args = ap.parse_args()

    self_test()

    products, goal_asins_in_order, human_ins = load_corpus(
        args.data_dir, args.num_products)
    goals = build_goals(products, goal_asins_in_order, human_ins)
    for i, g in enumerate(goals):
        g['query_id'] = i

    # official split buckets (baseline_models/env.py)
    buckets = {'test': goals[0:500], 'dev': goals[500:1500],
               'train': goals[1500:]}
    rng_sample = random.Random(20260922)
    sampled = {}
    for split, n in [('train', args.n_train), ('dev', args.n_dev),
                     ('test', args.n_test)]:
        bucket = buckets[split]
        idxs = sorted(rng_sample.sample(range(len(bucket)), min(n, len(bucket))))
        sampled[split] = [bucket[i] for i in idxs]
        print(f"[sample] {split}: {len(sampled[split])} queries from official "
              f"bucket (size {len(bucket)})")

    # BM25 index over the product subset (official indexed fields)
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

    query_pool = [products[a]['query'] for a in asins[:20000]
                  if products[a]['query']]

    eps_mix = {'train': [0.0, 0.0, 0.1, 0.2], 'dev': [0.1], 'test': [0.0]}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for split, goal_list in sampled.items():
        rows, rewards = [], []
        for i, goal in enumerate(goal_list):
            eps = eps_mix[split][i % len(eps_mix[split])]
            ep, reward = run_episode(goal, products, bm25, asins, split, i,
                                     eps, query_pool)
            validate(ep)
            rows.append(ep)
            rewards.append(reward)
        path = out_dir / f"webshop_{split}.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_cands = [len(s['candidates']) for r in rows for s in r['steps']]
        rewards_sorted = sorted(rewards)
        stats = {
            'episodes': len(rows),
            'mean_steps': round(sum(r['n_steps'] for r in rows) / len(rows), 2),
            'mean_candidates': round(sum(n_cands) / len(n_cands), 2),
            'reward_mean': round(sum(rewards) / len(rewards), 4),
            'reward_p50': rewards_sorted[len(rewards) // 2],
            'reward_min': rewards_sorted[0],
            'success_rate': round(sum(r['success'] for r in rows) / len(rows), 4),
        }
        all_stats[split] = stats
        print(f"[write] {path} | {stats}")
    qids = {s: {g['query_id'] for g in sampled[s]} for s in sampled}
    assert not (qids['train'] & qids['dev']) and not (qids['train'] & qids['test']) \
        and not (qids['dev'] & qids['test']), "splits must be disjoint"
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    print("[done] stats:", json.dumps(all_stats))


if __name__ == '__main__':
    main()
