# ALFWorld Benchmark (Long-Jev unified episode format)

TextWorld text-household tasks (ALFRED-derived): put / clean / heat / cool / examine
under lamp / pick-two. Each episode is one complete **expert trajectory** with
per-step observation, multiple-choice candidates (gold + distractors), gold action
and a short event string. Generated 2026-09-22 by `code/memexp/gen_data_alfworld.py`.

## Files

| file | episodes | source split |
|---|---|---|
| `alfworld_train.jsonl` | 400 | ALFWorld train games (pool 3,553) |
| `alfworld_dev.jsonl`   | 50  | ALFWorld valid_seen (pool 140) |
| `alfworld_test.jsonl`  | 100 | ALFWorld valid_unseen (pool 134) — official generalization split (unseen scenes) |

`train.jsonl` / `dev.jsonl` / `test.jsonl` are symlinks to the same files.
Raw data + provenance in `raw/` (see below). Splits share no game files (checked).

## Sources & license

- **Official ALFWorld release 0.2.2** (github.com/alfworld/alfworld, MIT license,
  Copyright (c) 2020 Mohit Shridhar and others):
  - `raw/json_2.1.1_json.zip` (72.0 MB) — ALFRED `traj_data.json` per trial
    (train 6,374 / valid_seen 251 / valid_unseen 477 trials), used for cross-checks.
  - `raw/json_2.1.1_pddl.zip` (34.9 MB) — `initial_state.pddl` per trial.
  - Canonical URLs (from `scripts/alfworld-download` in the repo):
    `https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip`
    and `.../json_2.1.1_pddl.zip`. Downloaded via ghproxy mirror (gh.ddlc.top)
    because both the A6000 (campus captive portal) and the local machine cannot
    reach github.com directly.
- **Game files (`game.tw-pddl`)**: mirrored verbatim from HuggingFace dataset
  `awawa-agi/alfworld-raw` (parquets in `raw/`, 3,553/140/134 games = official
  solvable-game counts). Each game embeds the PDDL problem, the TextWorld grammar,
  and the **official engine-verified expert walkthrough** (grounded text commands
  with entity instance numbers). `raw/games_*.jsonl` are compact per-game extracts
  (produced by `raw/preprocess_parquets.py`); the tw-pddl game-engine package was
  NOT installed and no THOR rendering was used.

### Why reconstruct instead of running the engine
`pip install alfworld` requires compiling TextWorld (failed previously on this
machine; not pursued per plan). Instead the generator replays each official
walkthrough against a deterministic world model parsed from the game's PDDL
problem and renders observations with the **official `alfred.twl2` grammar
templates**. Verification:
- Demangling/numbering replicates `alfworld/agents/utils/misc.py::Demangler`
  (lowercase ids, sort, per-class pool numbers popped from the end; `Sink/Bathtub`
  + "basin" -> `sinkbasin`/`bathtubbasin`). Acid test: **all 140 valid_seen
  walkthroughs (812 commands) resolve 100%** against this numbering.
- Walkthrough verb sequences match the official `traj_data.json` `high_pddl`
  plans for 76% of sampled pick_and_place/look_at games (checked via the official
  zip); the differences are benign: solver ordering, and extra engine-required
  `open` before putting into closed drawers/cabinets (the TW walkthrough is the
  engine-correct variant).
- With the full PDDL precondition model (agent shares a *location* with the target
  receptacle; several receptacles share one location) **0 of 3,827 games are
  dropped** during replay.

## Episode schema (one JSON object per line)

```json
{"task":"alfworld","episode_id":"alfworld-<split>-<idx>","split":"train|dev|test",
 "meta":{"game_file":"<task_dir>/<trial_id>/game.tw-pddl","task_type":"<one of 6>"},
 "success":true,"n_steps":N,
 "steps":[{"t":1,"obs":"...","candidates":{"<cmd>":"<human desc>",...},
           "action":"<cmd>","event":"..."}]}
```

- `action` is always a key of the same step's `candidates`; candidates 3–8 per
  step (hard cap 10); dict order = presentation order (shuffled, gold not
  position-biased); keys are stable command strings.
- `obs` = the previous step's environment feedback (first step: game intro with
  room receptacle list and task). Max observed 907 chars (limit 3,500; the
  truncate-middle rule was never triggered, so **no obs were truncated**).
- `event` ≤ 200 chars: short result summary ("picked up the mug 1 from the desk 1.",
  "moved to the fridge 1; it is closed.").
- `success=true`: every episode ends with the walkthrough's final command, i.e.
  the engine-verified expert solution completed.
- Entity names carry instance numbers exactly as the official games render them
  (`countertop 1`, `sinkbasin 1`, `cd 2`).
- **json_2.1.1 command dialect** (as in the shipped games; differs from newer
  json_2.1.3/pip wording): putting is `move X to Y` (not `put X in/on Y`), lamps
  are `use desklamp 1` (not `toggle`), articles are always "a" (`a egg 1`).

## Distractor candidate construction

At each step the context is: receptacles sharing the agent's location (per PDDL
`receptacleAtLocation`), their visible contents (open receptacles must be open),
the held object, and room receptacle list from the intro. The candidate pool is
instantiated from ALFWorld command templates exactly where the engine's
preconditions make them legal:

- `go to {recep}` for every receptacle not co-located with the agent;
- `take {obj} from {recep}` for every visible non-lamp object on co-located
  receptacles (when hands are empty);
- `move {held} to {recep}` for co-located receptacles (when holding);
- `open`/`close {recep}` for co-located openable receptacles (state-appropriate);
- `heat {held} with microwave {n}` / `cool ... with fridge {n}` /
  `clean ... with sinkbasin|bathtubbasin {n}` when at that appliance and holding;
- `use {desklamp|floorlamp} {n}` when the lamp is on a co-located receptacle;
- `examine {recep}` / `examine {held}`.

Gold + 4–7 distractors are sampled from this pool with
`random.Random(f"{seed}|{split}|{idx}|{t}")` (fully deterministic; seed 13),
then shuffled for presentation. Each candidate's desc is a human phrasing
("go to the countertop 1", "turn on the desklamp 1"). If a pool is too small the
episode keeps fewer candidates honestly (minimum observed in practice: 5).

## Statistics (generated on A6000, validated by an independent checker)

| split | episodes | steps mean (min–max) | cand/step mean | task types |
|---|---|---|---|---|
| train | 400 | 6.14 (3–10) | 6.52 | pick_two 108, pick&place 90, clean 78, cool 61, heat 40, look 23 |
| dev   | 50  | 5.60 (3–9)  | 6.40 | pick&place 14, clean 11, cool 10, pick_two 8, look 6, heat 1 |
| test  | 100 | 5.89 (3–9)  | 6.49 | clean 25, pick&place 18, cool 18, heat 16, look 12, pick_two 11 |

Step-count distribution (train): 1–3: 11, 4–6: 224, 7–9: 163, 10+: 2.
obs mean 144 chars; event mean 44 chars. Validation asserts: episode ids unique and
well-formed, action ∈ candidates, 3 ≤ |candidates| ≤ 10, obs ≤ 3,500 chars,
event ≤ 200 chars, success=true, no cross-split game overlap — all pass.

md5: train `0ed0e439fc3d8e3c832eab8c6a28f9f8`, dev `9136bd0b1f0f9c6d62e5d07850a20e71`,
test `021ea0e2d23b37a3f61856b1de5bd3f5`.

## Sample line (first line of alfworld_test.jsonl, abridged obs)

```json
{"task": "alfworld", "episode_id": "alfworld-test-0", "split": "test", "meta": {"game_file": "look_at_obj_in_light-Mug-None-DeskLamp-308/trial_T20190908_201421_021646/game.tw-pddl", "task_type": "look_at_obj_in_light"}, "success": true, "n_steps": 3, "steps": [{"t": 1, "obs": "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a desk 2, a drawer 1, ..., and a shelf 6.\n\nYour task is to: examine the mug with the desklamp.", "candidates": {"go to drawer 4": "go to the drawer 4", "go to shelf 5": "go to the shelf 5", "go to desk 1": "go to the desk 1", ...}, "action": "go to desk 1", "event": "moved to the desk 1; on it you see 3 objects."}, {"t": 2, "obs": "You arrive at desk 1. On the desk 1, you see a creditcard 3, a desklamp 1, a laptop 2, a mug 1, a pen 1, and a pencil 1.", "candidates": {..., "take pen 1 from desk 1": "take the pen 1 from the desk 1", "use desklamp 1": "turn on the desklamp 1"}, "action": "use desklamp 1", "event": "turned on the desklamp 1."}, {"t": 3, "obs": "You turn on the desklamp 1.", "candidates": {..., "take laptop 2 from desk 1": "take the laptop 2 from the desk 1", ...}, "action": "take mug 1 from desk 1", "event": "picked up the mug 1 from the desk 1."}]}
```

## Known caveats (honest notes)

- Expert walkthroughs are concise: 3–10 steps (mean ~6), not 5–50; longer
  horizons in ALFWorld come from exploration under imperfect knowledge, not from
  longer expert paths.
- Obs/feedback strings are rendered from the official grammar templates by our
  replay, not captured from a live engine run; intro receptacle order is our
  deterministic sorted order (the engine's own iteration order is unspecified).
- The official PDDL init sometimes lists one object in several receptacles
  (~2% of facts, 39% of games affected at least once); we keep all placements
  (as a fact set, like TextWorld) so the object can be seen (and taken) at each.
- The raw-id walkthroughs shipped for the 134 valid_unseen games were
  re-demangled with the same rule as valid_seen (verified: every id resolves).
- `raw/` also contains the two official zips (provenance / cross-checks),
  the three source parquets, `preprocess_parquets.py`, and the compact
  `games_{train,valid_seen,valid_unseen}.jsonl` consumed by the generator.

## 全量版（2026-09-22 重生成，覆盖 400-ep 冒烟版；旧版保留为 *_s400.jsonl）

官方全量：train=3,553 / dev=valid_seen 140 / test=valid_unseen 134（官方泛化切分全量）

- `alfworld_train.jsonl`: 3553 episodes / 21335 决策点 / success=1.000
- `alfworld_dev.jsonl`: 140 episodes / 812 决策点 / success=1.000
- `alfworld_test.jsonl`: 134 episodes / 788 决策点 / success=1.000

## 官方划分对齐声明

对齐对象：ALFWorld 官方 json_2.1.1 release（Shridhar et al., 2020）。官方划分 train / valid_seen / valid_unseen，本目录采用官方全量，映射 dev=valid_seen、test=valid_unseen（论文标准泛化评测切分），无自造切分、无丢弃（3,827 局全部转换）：
- train=官方 train 全量 3553 episodes / 21335 决策点
- dev=valid_seen 官方全量 140 episodes / 812 决策点
- test=valid_unseen 官方全量 134 episodes / 788 决策点
