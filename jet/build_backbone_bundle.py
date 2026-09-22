"""Build a decision-model bundle from any Qwen3 backbone with a fresh head.

Mirrors train_toy_decisions.py's construction (AutoModelForCausalLM fp32 ->
DecisionModel(lm.model, set_head)) but skips the toy-data training: the head is
randomly initialized (seeded), the backbone carries pretrained weights. The
resulting bundle loads unchanged in DecisionPredictor / train_generic /
train_mem_generic via --checkpoint.
"""
import argparse
import json
import random
from pathlib import Path

import torch

from jet.vendor.predict_toy_decisions import DecisionModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="local dir or HF name of the Qwen3 backbone")
    p.add_argument("--out", required=True)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--set-head", choices=["none", "attention"], default="attention")
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32,
                                              attn_implementation="sdpa")
    lm.config.use_cache = False
    model = DecisionModel(lm.model, a.set_head)
    n_params = sum(t.numel() for t in model.parameters())
    del lm

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"model": a.model, "set_head": a.set_head, "max_length": a.max_length,
              "seed": a.seed, "parameter_count": n_params,
              "parameter_storage": "float32", "forward_autocast": "bfloat16",
              "note": "pretrained backbone + freshly initialized head; built by build_backbone_bundle.py"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    tokenizer.save_pretrained(out / "tokenizer")
    model.backbone.config.save_pretrained(out / "backbone_config")
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
              out / "best.safetensors")
    print(json.dumps({"out": str(out), "params": n_params,
                      "hidden": model.backbone.config.hidden_size,
                      "layers": model.backbone.config.num_hidden_layers,
                      "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id}))


if __name__ == "__main__":
    main()
