# Model checkpoints

Place the [NanoJev](https://github.com/TianyuCodings/NanoJev) 0.6B decision
model bundle in `checkpoints/NanoJev-unified/` before training. The bundle must
contain `config.json`, `best.safetensors`, `tokenizer/`, and
`backbone_config/`.

Jeτ training writes task-specific bundles to `checkpoints/jet/jet06_<task>/`.
The trained bundle contains the same four components plus a `writer.*` latent
state module in `best.safetensors` and a `train_log.json` file. The `jet/`
subdirectory is ignored by Git because model weights are distributed
separately.

The repository does not include model weights. See the root README for the
training and evaluation commands.
