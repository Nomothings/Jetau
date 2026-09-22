# WebShop benchmark episodes (Long-Jev unified episode JSONL)

Offline-generated WebShop shopping episodes in the unified Long-Jev episode
format, for the parallel-decision long-horizon memory experiments
(NanoJev / Qwen3-0.6B, arms C1/C3/C4).

**This is an offline faithful replica: official WebShop data subset + offline
BM25 retrieval replica + official reward formula.** No WebShop server, Java or
Lucene is involved; everything is reproducible from the raw files in `raw/`
with one deterministic CPU-only script.

## Files

| file | episodes | notes |
|---|---|---|
| `webshop_train.jsonl` | 400 | teacher with epsilon-exploration mix {0, 0, 0.1, 0.2} |
| `webshop_dev.jsonl` | 60 | teacher with epsilon = 0.1 |
| `webshop_test.jsonl` | 100 | pure teacher (epsilon = 0) |
| `stats.json` | — | per-split statistics dumped by the generator |
| `raw/` | — | official data files (see provenance below) |
| `generate.log` | — | log of the generation run on this machine |

Generator: `/data/yangyuming/Long-Jev/code/memexp/gen_data_webshop.py`
(runner: `run_webshop_gen.sh` in the same directory).

## Data source and license

- Code and data semantics: the official repo `princeton-nlp/WebShop`
  (branch `master`), **MIT License**, Copyright (c) 2023 Princeton Natural
  Language Processing. Source tarball fetched via the `ghfast.top` GitHub
  mirror (A6000 and direct GitHub were unreachable, see "Pitfalls" in the
  handover notes).
- Product / instruction data: HF dataset mirror `YWZBrandon/webshop-data`
  (files `items_shuffle.json`, `items_ins_v2.json`, `items_human_ins.json`,
  2025-09-01 version), which matches the files the official `setup.sh`
  downloads from Google Drive (same filenames; `items_human_ins.json` is
  md5-identical to the copy bundled in the official repo at
  `baseline_models/data/items_human_ins.json`).
  - `items_shuffle.json`: 1,181,436 raw products (5.48 GB)
  - `items_ins_v2.json`: attributes for 1,181,436 ASINs
  - `items_human_ins.json`: human-written instructions for 10,136 ASINs
    (12,087 usable instructions; 164 skipped for empty attributes — same
    count as the official `human_goals.json`)

## What the replica does

1. **Corpus**: the full `items_shuffle.json` is scanned once (needed to
   recover the official goal ordering); the searchable catalog is a
   **59,680-product subset** (~50k requested): *all* 10,136 products that
   carry human instructions, plus file-order fillers up to the cap. The file
   is pre-shuffled by the official pipeline, so the fillers are a uniform
   sample. Retrieval therefore runs over ~50k products instead of 1.18M —
   ranking/difficulty differs slightly from the full official setting.
2. **Goals and official split**: faithful port of
   `web_agent_site/engine/goal.py::get_human_goals` + `SimServer`
   (`web_agent_site/envs/web_agent_text_env.py`): goals are built in
   product-file order, then `random.Random(233).shuffle(goals)`, then split
   per `baseline_models/env.py`: **test = goals[0:500], dev = goals[500:1500],
   train = goals[1500:]** (bucket sizes 500 / 1,000 / 10,587). The 400/60/100
   queries are deterministic samples (`random.Random(20260922)`) from those
   official buckets; `meta.query_id` is the goal's index in the official
   shuffled list. Splits are disjoint by construction (asserted).
3. **Search**: BM25 (k1=1.5, b=0.75, Lucene-style idf) over exactly the
   fields the official Lucene index uses (Title + Description +
   BulletPoints[0] + option text, lowercased alphanumeric tokens); top-50
   like the official `SEARCH_RETURN_N = 50`; results page shows 10.
4. **Actions / candidates** (3–8 per step, hard cap 10, dict order =
   presentation order, keys stable):
   - `search[<query>]` — issue a search query
   - `click[item[B07XXXXXXX]]` — open a result item
   - `click[option[<name>: <value>]]` — select a product option
   - `buy[now]` — purchase the currently open item
5. **Teacher**: rule shopper maximising the official reward:
   search instruction keywords → refine with goal-title keywords if the goal
   is not on the page (≤3 searches) → open the goal item → select the goal
   option values → `buy[now]`; if the goal is unreachable it commits to the
   displayed item with the highest official reward (anti-oscillation).
   Small epsilon exploration is mixed into train/dev (see table above).
   Every episode terminates with a purchase (step budget 8; forced buy if
   exhausted, flagged in the event text).
6. **Reward (exact port of `goal.py::get_reward`)**:
   `total = (attr_matches + option_matches + price_ok) / (n_goal_attrs + n_goal_options + 1) * r_type`
   with `r_type` from query/category/title matching.
   `success = reward >= 0.7` (threshold fixed by the task card).

### Documented deviations from the official environment

- **r_type title_score**: official uses spaCy noun extraction; offline we use
  lowercase-token overlap minus stopwords. `query_match` / `category_match`
  are exact. For teacher purchases of the goal product `query_match` is
  always True, so `r_type = 1` exactly as official; the approximation only
  affects exploration purchases of other products.
- **Fuzzy matching** (`thefuzz.token_set_ratio` for attributes/options) is
  re-implemented with `difflib.SequenceMatcher` — the same fallback path
  `thefuzz` itself uses. All official unit-test cases from
  `tests/web-agent-site/engine/test_goal.py` that do not need spaCy pass
  (asserted at generator start), including the fuzzy ones
  ("essential oil" vs "essential oils" = 96 > 85, "tea bag" vs "tea tree" =
  60 ≤ 85, "powder snow" vs "cool powder snow" = match).
- **Price sampling**: official samples range prices and instruction price
  ceilings from the *unseeded* global RNG (i.e. nondeterministic in the
  official env); we sample from the same `PRICE_RANGE` procedure with
  per-item md5 seeds, so runs are deterministic.
- **Retrieval pool** is the ~50k subset (point 1 above).
- Instruction/option/attribute data and the reward formula itself are
  unmodified official data and code.

## Statistics (A6000 run, 2026-09-22)

| split | episodes | mean steps | mean candidates/step | reward mean | reward p50 | success rate |
|---|---|---|---|---|---|---|
| train | 400 | 4.63 | 5.92 | 0.890 | 1.00 | 84.8% |
| dev   | 60  | 4.90 | 5.81 | 0.867 | 1.00 | 81.7% |
| test  | 100 | 4.48 | 5.98 | 0.983 | 1.00 | 99.0% |

Schema validation (asserted per episode in the generator): `action ∈
candidates` at every step; 1–10 candidates per step; `obs ≤ 3500` chars
(observed max 1956, well under the 2048-token training-sample bound);
`event ≤ 200` chars (observed max 107); `episode_id` matches split;
split query-id sets pairwise disjoint; JSON round-trips.

Pure-teacher test failures (1/100) are honest hard queries where the goal
item does not enter the top-7 candidates within the search budget.

## Regenerate

```bash
# on the A6000 box (data already in raw/)
bash /data/yangyuming/Long-Jev/code/memexp/run_webshop_gen.sh
# which runs:
/data/yangyuming/Long-Jev/code/memexp/.venv/bin/python \
    /data/yangyuming/Long-Jev/code/memexp/gen_data_webshop.py \
    --data-dir /data/yangyuming/Long-Jev/data/benchmarks/webshop/raw \
    --out-dir /data/yangyuming/Long-Jev/data/benchmarks/webshop \
    --num-products 50000 --n-train 400 --n-dev 60 --n-test 100
```

Stdlib only (no pip installs needed), ~10 min, ~40 GB RAM peak (full
`items_shuffle.json` is loaded once), no GPU. The pipeline is fully
deterministic: an independent rerun on another machine produced
byte-identical JSONL (md5-verified).

## Sample episode (one line of `webshop_train.jsonl`)

```json
{"task": "webshop", "episode_id": "webshop-train-0293", "split": "train", "meta": {"query_id": 9122, "instruction": "i am looking fresh scent body lotion for dry skin, and price lower than 40.00 dollars"}, "success": true, "n_steps": 4, "steps": [{"t": 1, "obs": "Instruction: i am looking fresh scent body lotion for dry skin, and price lower than 40.00 dollars\nWebShop search page. Type a search query to find the product you need, then buy the right item with the right options.", "candidates": {"search[fresh scent body lotion dry skin]": "search for 'fresh scent body lotion dry skin'", "search[body care]": "search for 'body care'", "search[baby foods]": "search for 'baby foods'", "search[men's sandals]": "search for 'men's sandals'", "search[television accessories]": "search for 'television accessories'"}, "action": "search[fresh scent body lotion dry skin]", "event": "searched for 'fresh scent body lotion dry skin'; 50 results."}, {"t": 2, "obs": "Instruction: i am looking fresh scent body lotion for dry skin, and price lower than 40.00 dollars\nSearch results for 'fresh scent body lotion dry skin' (showing top 10 of 50 matches):\n1. B00DEX61A8 | NIVEA Men Maximum Hydration 3 in 1 Nourishing Lotion 16.9 Fl Oz | $5.48\n2. B077VQD17T | Hempz Herbal Body Moisturizer for Women with 100% Pure Hemp Seed Oil, Sugarcane & Papaya, 17 fl. oz. - Moistu… | $15.37\n3. B0091JI3YG | Avon Skin So Soft Original Body Lotion with Jojoba - 11.8 oz | $9.5\n4. B087Y28XLM | Palmer's Mens Body/face Lotion 8.5 Ounce | $5.99\n5. B07X1FVJP6 | Natural Shea Butter Body Lotion Stick with Coconut Oil and Beeswax, in a 2 oz. Solid Lotion Bar (Lemongrass L… | $11.97\n6. B0746SNMQD | Diva stuff Pre-Swim Aqua Therapy Chlorine Neutralizing Body Moisturizing Lotion for Swimmers, Protects Skin f… | $13.99\n7. B096T9KTSD | TRIHARD's After Swim Body Wash and Swimmers Shampoo Extra Boost, Pre & Post Swim Conditioner and Body Lotion… | $73.35\n8. B07KDX6TJN | Live Clean Fresh Water Hydrating Body Lotion, 17 oz Each bottle (2-pack) | $23.84\n9. B001459IEE | Aveeno Daily Moisturizing Body Lotion with Soothing Oat and Rich Emollients to Nourish Dry Skin, Gentle & Fra… | $8.99\n10. B08YXTHGVC | Victoria's Secret Coconut Passion Refreshing Gel Body Wash (Coconut Passion) | $16.69", "candidates": {"click[item[B00DEX61A8]]": "open item: NIVEA Men Maximum Hydration 3 in 1 Nourishing Lotion 16.9 F… ($5.48)", "click[item[B077VQD17T]]": "open item: Hempz Herbal Body Moisturizer for Women with 100% Pure Hemp… ($15.37)", "click[item[B0091JI3YG]]": "open item: Avon Skin So Soft Original Body Lotion with Jojoba - 11.8 oz ($9.5)", "click[item[B087Y28XLM]]": "open item: Palmer's Mens Body/face Lotion 8.5 Ounce ($5.99)", "click[item[B07X1FVJP6]]": "open item: Natural Shea Butter Body Lotion Stick with Coconut Oil and… ($11.97)", "click[item[B0746SNMQD]]": "open item: Diva stuff Pre-Swim Aqua Therapy Chlorine Neutralizing Body… ($13.99)", "click[item[B096T9KTSD]]": "open item: TRIHARD's After Swim Body Wash and Swimmers Shampoo Extra B… ($73.35)", "search[palmer s mens body face lotion 8 5]": "search for 'palmer s mens body face lotion 8 5' instead"}, "action": "click[item[B087Y28XLM]]", "event": "opened item B087Y28XLM: Palmer's Mens Body/face Lotion 8.5 Ounce."}, {"t": 3, "obs": "Instruction: i am looking fresh scent body lotion for dry skin, and price lower than 40.00 dollars\nItem: Palmer's Mens Body/face Lotion 8.5 Ounce\nPrice: $5.99\nCategory: Beauty & Personal Care › Skin Care › Body › Moisturizers › Lotions\nAttributes: animal testing; dry skin; paraben free\nOptions:\n- scent: fresh, butter\n- size: 8.5 fl oz (pack of 1), 10.6 fl oz (pack of 1)\nDescription: Palmers men's Body/face Ltn 8.5 Oz.", "candidates": {"click[option[scent: fresh]]": "choose scent 'fresh'", "click[option[scent: butter]]": "choose scent 'butter'", "buy[now]": "buy this item with current options", "search[palmer s mens body face lotion 8 5]": "go back and search again"}, "action": "click[option[scent: fresh]]", "event": "chose option 'scent: fresh'."}, {"t": 4, "obs": "Instruction: i am looking fresh scent body lotion for dry skin, and price lower than 40.00 dollars\nItem: Palmer's Mens Body/face Lotion 8.5 Ounce\nPrice: $5.99\nCategory: Beauty & Personal Care › Skin Care › Body › Moisturizers › Lotions\nAttributes: animal testing; dry skin; paraben free\nOptions:\n- scent: fresh, butter\n- size: 8.5 fl oz (pack of 1), 10.6 fl oz (pack of 1)\nDescription: Palmers men's Body/face Ltn 8.5 Oz.", "candidates": {"buy[now]": "buy this item now ($5.99)", "click[item[B00DEX61A8]]": "open item: NIVEA Men Maximum Hydration 3 in 1 Nourishing Lot… instead", "search[palmer s mens body face lotion 8 5]": "search again for the requested product"}, "action": "buy[now]", "event": "bought item B087Y28XLM (Palmer's Mens Body/face Lotion 8.5 Ounce); reward 1.00."}]}
```

(Verbatim line from the generated file; single complete JSON object,
4859 characters.)

## 全量版（2026-09-22 重生成，覆盖 400-ep 冒烟版；旧版保留为 *_s400.jsonl）

官方全量：train=10,587 / dev=1,000 / test=500（官方 Random(233) 划分规则，12,087 条人类指令全量）

- `webshop_train.jsonl`: 10587 episodes / 49816 决策点 / success=0.861
- `webshop_dev.jsonl`: 1000 episodes / 4794 决策点 / success=0.818
- `webshop_test.jsonl`: 500 episodes / 2268 决策点 / success=0.968

## 官方划分对齐声明

对齐对象：WebShop 官方 repo `baseline_models/env.py` 的划分规则——12,087 条人类指令按文件序构建后 `random.Random(233).shuffle`，test=[0:500)、dev=[500:1500)、train=[1500:]。本目录逐行复刻该规则并使用**全部三个桶的全量**（10,587/1,000/500），query_id 保留官方全序索引；唯一偏差为价格采样的确定化与 title_score 启发式（不影响划分与桶成员，见上文偏差说明）。
- train 10587 episodes / 49816 决策点
- dev=官方dev桶 1000 episodes / 4794 决策点
- test=官方test桶 500 episodes / 2268 决策点
