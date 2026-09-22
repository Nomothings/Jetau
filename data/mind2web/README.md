# Mind2Web benchmark (Long-Jev unified episode format)

Raw data + converted unified-format episode JSONLs (`train.jsonl`, `dev.jsonl`,
`test.jsonl`). Generation is fully deterministic (seed 17); reruns are
byte-identical.

## Raw data

Source: official [osunlp/Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web)
release, downloaded from `hf-mirror.com` (via the machine's local proxy), same
route the earlier `train_10.json` came from.

| file | what |
|---|---|
| `train_0.json` … `train_4.json` | official train split shards (~100 instances each, ~600 MB each; downloaded 2026-09-22 for this conversion because `train_10.json` alone holds only 9 instances) |
| `train_10.json` | tail shard of the train split, 9 instances (28 MB) |
| `test.zip` → `test_domain/`, `test_task/`, `test_website/` | official test splits (cross-domain / cross-task / cross-website) |
| `test_domain/test_domain_0.json` | shard 0 of the cross-domain test split, **exactly 100 instances** |

Structure warning (this bit an earlier eval): these files are **top-level JSON
arrays** of instance dicts, mostly serialized compactly on one line. Parsing
them line-by-line as JSONL makes `rows[0]` the whole list and crashes with
`AttributeError: 'list' object has no attribute 'get'` (see
`nanojev-memexp/memexp/logs/mind2web_test.log`). Always stream-parse the array.
Per-instance fields: `annotation_id` (unique instance id), `website`, `domain`,
`subdomain`, `confirmed_task`, `action_reprs` (row-level, e.g.
`"[link]  NFL . -> CLICK"`), `actions[]` with `action_uid`,
`operation{op,value}`, `pos_candidates[]`, `neg_candidates[]`,
`raw_html`/`cleaned_html`. Candidates carry only `tag`, `backend_node_id` and
`attributes` (a JSON-encoded string; visible text must be extracted from keys
like `aria_label`/`title`/`placeholder`/`input_value`/`value`/`alt`).

## Unified episode format

One episode per line = one annotation instance (all of its action steps).
`success: true` (expert trajectories).

```json
{"task":"mind2web","episode_id":"mind2web-<split>-<idx>","split":"train|dev|test",
 "meta":{"website":"budget","instance_id":"<annotation_id>","confirmed_task":"...",
         "domain":"Travel","subdomain":"Car rental","truncated":false,"dropped_steps":0},
 "success":true,"n_steps":6,
 "steps":[{"t":1,
           "obs":"Website: budget (Travel/Car rental).\nTask: ...\nStep 1 of 6\nPrevious action: none (first step)\nChoose the element to interact with next.",
           "candidates":{"option_1":"footer | class=footer","option_4":"a | role=button | class=dropdown-toggle", "...":"..."},
           "action":"option_4","event":"clicked [button] Reservations."}]}
```

Actual first line of `train.jsonl` (real example, one episode):

```json
{"task": "mind2web", "episode_id": "mind2web-train-0000", "split": "train", "meta": {"website": "budget", "instance_id": "b02e47ac-c1a6-4f5c-886f-f32af745e7f3", "confirmed_task": "View a reservation made under the last name Walker in Australia for a car using the reservation confirmation number A987654.", "domain": "Travel", "subdomain": "Car rental", "truncated": false, "dropped_steps": 1}, "success": true, "n_steps": 6, "steps": [{"t": 1, "obs": "Website: budget (Travel/Car rental).\nTask: View a reservation made under the last name Walker in Australia for a car using the reservation confirmation number A987654.\nStep 1 of 6\nPrevious action: none (first step)\nChoose the element to interact with next.", "candidates": {"option_1": "footer | class=footer", "option_2": "a", "option_3": "body | class=INDlangdirLTR INDpositionRight INDChrom", "option_4": "a | role=button | class=dropdown-toggle", "option_5": "div", "option_6": "li", "option_7": "ul | class=header-secondary"}, "action": "option_4", "event": "clicked [button] Reservations."}, {"t": 2, "obs": "Website: budget (Travel/Car rental).\nTask: View a reservation made under the last name Walker in Australia for a car using the reservation confirmation number A987654.\nStep 2 of 6\nPrevious action: [button] Reservations -> CLICK\nChoose the element to interact with next.", "candidates": {"option_1": "a | class=footer-header arrow-down-grey-mob", "option_2": "li", "option_3": "div | class=special-promo-content", "option_4": "div | class=special-promo special-promo-right blue-promo hidde", "option_5": "a", "option_6": "li | class=dropdown", "option_7": "div | id=INDquickAccess"}, "action": "option_5", "event": "clicked [link] View / Modify / Cancel."}]}
```

### Field construction rules

- **candidates**: `1 positive + up to 6 sampled negatives` (hard range 2–10;
  minimum honest floor is 2 = 1 pos + 1 neg). In these three files every step
  ended up with exactly 7 candidates because raw Mind2Web actions carry
  hundreds of negatives.
- **negative sampling & order (deterministic)**: per step,
  `rng = random.Random(f"{seed}:{action_uid}")`; the index list of that
  action's `neg_candidates` is shuffled with this rng, negatives are taken in
  that order and deduped by `backend_node_id` **and** by final desc string
  (a negative whose desc would equal the positive's is dropped); then the
  combined `[pos] + negs` list is shuffled once more with the same rng, so the
  positive's position is uniform (empirically uniform over slots 1–7), never
  fixed first. `action` = key of the positive (`option_k`), keys are assigned
  in presentation order.
- **candidate desc** (style of `eval_mind2web.cand_desc`, adapted to the raw
  release where text lives inside `attributes`): `tag | id=<id/name> |
  role=<role/type> | text: <first non-empty of aria_label/title/placeholder/
  input_value/value/text_value/alt/name, ≤120 chars> | class=<≤50 chars>`,
  empty parts omitted. Many raw elements genuinely have none of these keys, so
  some candidates are bare `tag | class=...` — that is the honest information
  content of the raw file (no page HTML is put in `obs` by design).
- **obs** (≤3500 chars, step-local): website (+ domain/subdomain), task
  (`confirmed_task`), `Step t of N`, previous executed `action_repr` (or
  "none (first step)"). Longer history is the memory arm's job.
- **event** (≤200 chars): derived from the `action_repr` suffix after `-> `:
  `clicked [button] Reservations.` / `typed 'Walker' into [input].` /
  `selected 'AUSTRALIA' in [combobox] Select Residency.` / `hovered over ...`.

### Split semantics

- **train / dev**: both from the official **train** split pool
  (`train_0..4.json` + `train_10.json`, 509 instances total). Pool is sorted by
  `annotation_id`, shuffled once with `random.Random(17)`; the first 460 valid
  instances are materialized, **train = first 400, dev = next 60** (disjoint,
  verified: instance-id overlap 0).
- **test**: the official **cross-domain** split, shard
  `test_domain/test_domain_0.json` — its websites (reddit, udemy,
  theweathernetwork, apartments, ohio.gov, ...) are held-out domains that do
  not occur in the train split. The shard contains exactly 100 instances, so
  `test.jsonl` is the whole shard (order shuffled by the same seed).

### Dropped steps and truncation (honesty notes)

- Raw actions with **0 positive candidates** (~6–10% of actions; an annotation
  quirk) or 0 negatives cannot form a valid choice and are **dropped from the
  episode** (kept steps renumbered; count per episode in
  `meta.dropped_steps`). Totals: train 142, dev 42, test 42 steps. No instance
  was skipped entirely.
- Instances with >30 steps are truncated to the first 30 steps
  (`meta.truncated`). **Not triggered once** in these splits (max observed
  episode length < 30), so no episode here is truncated.

## Statistics

| split | file | episodes | steps | mean steps | mean cands/step | dropped steps |
|---|---|---|---|---|---|---|
| train | `train.jsonl` | 400 | 2855 | 7.14 | 7.00 | 142 |
| dev | `dev.jsonl` | 60 | 423 | 7.05 | 7.00 | 42 |
| test | `test.jsonl` | 100 | 618 | 6.18 | 7.00 | 42 |

Steps-per-episode histogram (bucket = [5k, 5k+4]):

| split | 0-4 | 5-9 | 10-14 | 15-19 | 20-24 | 25-29 | 30 |
|---|---|---|---|---|---|---|---|
| train | 143 | 165 | 61 | 22 | 4 | 5 | 0 |
| dev | 14 | 36 | 5 | 4 | 1 | 0 | 0 |
| test | 41 | 42 | 13 | 4 | 0 | 0 | 0 |

Website distribution top5 — train: spothero 15, budget 14, imdb 12,
ticketcenter 12, resy 11; dev: spothero 5, sixflags 4, new.mta.info 3,
kayak 3, gamestop 3; test: reddit 9, udemy 8, theweathernetwork 7,
apartments 7, ohio.gov 7.

## Reproduce

```bash
cd /data/yangyuming/Long-Jev/code/memexp
B=/data/yangyuming/Long-Jev/data/benchmarks/mind2web
.venv/bin/python gen_data_mind2web.py --input $B/train_{0,1,2,3,4,10}.json \
    --split train --episodes 400 --heldout 60 --seed 17 --out $B/train.jsonl
.venv/bin/python gen_data_mind2web.py --input $B/train_{0,1,2,3,4,10}.json \
    --split dev --episodes 60 --skip 400 --seed 17 --out $B/dev.jsonl
.venv/bin/python gen_data_mind2web.py --input $B/test_domain/test_domain_0.json \
    --split test --episodes 100 --seed 17 --out $B/test.jsonl
.venv/bin/python validate_mind2web.py   # schema + stats + disjointness checks
```

Large input files are read incrementally (`json.JSONDecoder.raw_decode` over
4 MiB chunks), so ~600 MB shards never load fully into RAM.

## 官方划分对齐（2026-09-22 终版）

对齐对象：官方 dataset card（huggingface.co/datasets/osunlp/Mind2Web）与官方 GitHub 说明。
官方 release 划分：**train 1,009 实例；test = cross-task 252 / cross-website 177 / cross-domain 912；无官方 dev**。

我们的划分：
- train = 官方 train 全量 1,009 中的 959（官方无 dev，从官方 train 池确定性留出 50 实例作 dev，seed 17，skip 959）
- test = 官方 cross-domain **全量 912 实例**（10 个分片全部转换）；另附官方 cross-task 252、cross-website 177 两个切分文件
- 三个 test 切分的分片实例数已逐一核对：task 100+100+52=252、website 100+77=177、domain 9×100+12=912，与 dataset card 一致
- 转换按实例不丢弃（skipped_instances 为空）；个别步骤无 pos_candidates 的按步丢弃并计入 meta.dropped_steps
- 历史注：早期文档提到 1,165/276 为论文预发布口径，以 dataset card 的 1,009/912/252/177 为准

- `train.jsonl`: 959 episodes / 6997 决策点
- `dev.jsonl`: 50 episodes / 364 决策点
- `test.jsonl`: 912 episodes / 5590 决策点
- `test_task.jsonl`: 252 episodes / 1966 决策点
- `test_website.jsonl`: 177 episodes / 1314 决策点
