# Checkpoints

Model bundles are NOT committed to git (`.gitignore` keeps only the directory
skeleton). Two backbone bundles are expected here, plus the 12 trained Jeτ
bundles under `jet/`.

## Backbones

| Bundle | Size | Source | Place at |
|---|---|---|---|
| `NanoJev-unified` | 0.6B | Released NanoJev unified decision model (official NanoJev release; bundle layout: `config.json` + `backbone_config/` + `tokenizer/` + `best.safetensors`) | `checkpoints/NanoJev-unified/` |
| `Qwen3-1.7B-Base` | 1.7B | Hugging Face `Qwen/Qwen3-1.7B-Base` (plain causal-LM backbone) | `checkpoints/Qwen3-1.7B-Base/` |

The 1.7B decision model is built locally from the plain backbone with a fresh
(attention) head — pretrained backbone weights, seeded random head, no toy-data
training:

```bash
python -m jet.build_backbone_bundle --model checkpoints/Qwen3-1.7B-Base \
    --out checkpoints/decision-qwen3-1.7b
```

The resulting bundle loads unchanged as `--checkpoint checkpoints/decision-qwen3-1.7b`
in every trainer/evaluator.

## Trained Jeτ bundles (`jet/`)

`checkpoints/jet/` holds the 12 fleet bundles written by
`scripts/dispatch_jet_fleet.sh`:

```
jet/jet06_{maze,snake,pokemon,alfworld,mind2web,webshop}   # Jeτ on NanoJev-unified (0.6B)
jet/jet17_{maze,snake,pokemon,alfworld,mind2web,webshop}   # Jeτ on decision-qwen3-1.7b
```

Each bundle is a full decision-model snapshot (`config.json`, `backbone_config/`,
`tokenizer/`, `best.safetensors` incl. `writer.*` slot-head weights, plus
`train_log.json`). If they are not present, either rerun the fleet script or
copy them from the original experiment tree with `scripts/sync_artifacts.sh`.
