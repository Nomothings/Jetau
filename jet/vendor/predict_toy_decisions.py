#!/usr/bin/env python3
"""Batch predictions from a local decision-model checkpoint.

Vendored from NanoJev: DecisionModel (originally defined in
train_toy_decisions.py, which this file loaded as a sibling module) is
inlined below so this vendor module is self-contained."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def reject_nonfinite(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_request(payload):
    if not isinstance(payload, dict) or set(payload) != {"states"}:
        raise ValueError('Input must contain only {"states": [...]}')
    states = payload["states"]
    if not isinstance(states, list) or not states:
        raise ValueError("states must be a nonempty array")
    seen_ids = set()
    for state in states:
        if not isinstance(state, dict) or set(state) != {"id", "state", "questions"}:
            raise ValueError("Each state must contain only id, state, and questions")
        if not nonempty_text(state["id"]) or state["id"] in seen_ids:
            raise ValueError("state id must be a unique nonempty string")
        seen_ids.add(state["id"])
        # Keep state serialization aligned with the upstream trainer.
        if not isinstance(state["state"], (str, dict, list)):
            raise ValueError("state content must be a string, object, or array")
        if not state["state"]:
            raise ValueError("state content must not be empty")
        questions = state["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        for qid, question in questions.items():
            if not nonempty_text(qid) or not isinstance(question, dict):
                raise ValueError("question ID must be a nonempty string and its value an object")
            if set(question) - {"type", "instructions", "criteria"}:
                raise ValueError(f"{state['id']}:{qid} contains unsupported question fields")
            typ = question.get("type")
            if typ not in {"boolean", "choice", "score"} or not nonempty_text(question.get("instructions")):
                raise ValueError(f"{state['id']}:{qid} has an invalid type or instructions")
            if typ == "boolean":
                if "criteria" in question:
                    criteria = question["criteria"]
                    if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                        raise ValueError("Boolean criteria must contain only false and/or true")
                    if not all(nonempty_text(value) for value in criteria.values()):
                        raise ValueError("Boolean criterion must be a nonempty string")
            elif typ == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                    raise ValueError("Choice criteria must contain 2–255 entries")
                if not all(nonempty_text(k) and nonempty_text(v) for k, v in criteria.items()):
                    raise ValueError("Choice IDs and descriptions must be nonempty strings")
            else:
                criteria = question.get("criteria")
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                    raise ValueError("Score criteria must be an ordered array of 2–10 entries")
                if not all(nonempty_text(value) for value in criteria):
                    raise ValueError("Score descriptions must be nonempty strings")
    return states


def prepare_examples(payload, tokenizer, max_length):
    """Encode segments and candidate paths as in the upstream trainer."""
    states = validate_request(payload)
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if type(tokenizer.eos_token_id) is not int or tokenizer.eos_token_id < 0:
        raise ValueError("checkpoint tokenizer needs a valid eos_token_id")
    examples = []
    for row in states:
        for qid, q in row["questions"].items():
            typ = q["type"]
            if typ == "boolean":
                ids, texts = ["false", "true"], ["The proposition is true."]
            elif typ == "choice":
                ids = list(q["criteria"])
                texts = [f"{key}: {q['criteria'][key]}" for key in ids]
            else:
                ids = [str(i) for i in range(len(q["criteria"]))]
                texts = q["criteria"]
            segments = [f"State:\n{row['state']}\n",
                        f"Question type: {typ}\nQuestion:\n{q['instructions']}\n"]
            if typ == "boolean" and "criteria" in q:
                for key, label in (("false", "False"), ("true", "True")):
                    if key in q["criteria"]:
                        segments[1] += f"{label} criterion: {q['criteria'][key]}\n"
            prefix = sum([tokenizer.encode(t, add_special_tokens=False) for t in segments], [])
            leaves = [prefix + tokenizer.encode(f"Candidate:\n{t}\nDecision:", add_special_tokens=False)
                      + [tokenizer.eos_token_id] for t in texts]
            largest = max(map(len, leaves))
            if largest > max_length:
                raise ValueError(f"{row['id']}:{qid} candidate needs {largest} tokens, exceeding max_length={max_length}")
            examples.append({"id": f"{row['id']}:{qid}", "state_id": row["id"], "qid": qid,
                             "type": typ, "candidate_ids": ids, "candidate_texts": texts,
                             "leaf_tokens": leaves})
    return examples


def complete_question_batches(examples, batch_questions=0):
    if type(batch_questions) is not int or batch_questions < 0:
        raise ValueError("batch_questions must be nonnegative; 0 batches all questions")
    size = batch_questions or len(examples)
    if not examples:
        return []
    return [examples[i:i + size] for i in range(0, len(examples), size)]


def answer_from_probabilities(example, probabilities):
    ids = example["candidate_ids"]
    if len(probabilities) != len(ids) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError("Model produced invalid probabilities")
    if abs(math.fsum(probabilities) - 1.0) > 1e-5:
        raise ValueError("Model probabilities do not sum to one")
    best = max(range(len(ids)), key=probabilities.__getitem__)
    result = {"type": example["type"], "probabilities": dict(zip(ids, probabilities))}
    if example["type"] == "boolean":
        result.update(p_true=probabilities[1], value=bool(best))
    elif example["type"] == "choice":
        result.update(choice=ids[best], value=ids[best])
    else:
        score = math.fsum(i * p for i, p in enumerate(probabilities))
        result.update(score=score, level=best, value=score)
    return result


def local_checkpoint_files(checkpoint_dir):
    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("checkpoint-dir must be a local directory")
    paths = {"run_config": root / "config.json", "body_config": root / "backbone_config",
             "tokenizer": root / "tokenizer", "weights": root / "best.safetensors"}
    for label, path in paths.items():
        if not path.exists():
            raise ValueError(f"checkpoint is missing {label}: {path.name}")
    if not paths["run_config"].is_file() or not paths["weights"].is_file():
        raise ValueError("config.json and best.safetensors must be files")
    if not paths["body_config"].is_dir() or not paths["tokenizer"].is_dir():
        raise ValueError("backbone_config and tokenizer must be directories")
    return root, paths


class DecisionModel(nn.Module):
    """Copied verbatim from NanoJev train_toy_decisions.py (Jet vendored copy)."""

    def __init__(self, backbone, set_head):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        self.scalar = nn.Linear(hidden, 1)  # Nonzero random initialization avoids a dead first step.
        nn.init.normal_(self.scalar.weight, std=0.02)
        nn.init.zeros_(self.scalar.bias)
        self.set_head = set_head
        if set_head == 'attention':
            self.set_project = nn.Linear(hidden + 1, 128)
            self.set_attention = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
            self.set_output = nn.Linear(128, 1)
            # Only the final residual projection starts at zero; its upstream layers are nonzero.
            nn.init.zeros_(self.set_output.weight)
            nn.init.zeros_(self.set_output.bias)

    def forward(self, examples, pad_token):
        paths = [ids for ex in examples for ids in ex['leaf_tokens']]
        device = self.scalar.weight.device
        lengths = torch.tensor([len(ids) for ids in paths], device=device)
        width = int(lengths.max())
        tokens = torch.full((len(paths), width), pad_token, dtype=torch.long, device=device)
        for i, ids in enumerate(paths):
            tokens[i, :len(ids)] = torch.tensor(ids, device=device)
        attention = torch.arange(width, device=device)[None, :] < lengths[:, None]
        hidden = self.backbone(input_ids=tokens, attention_mask=attention,
                               use_cache=False).last_hidden_state
        leaves = hidden[torch.arange(len(paths), device=device), lengths-1]
        kmax = max(len(ex['candidate_ids']) for ex in examples)
        h = leaves.new_zeros((len(examples), kmax, leaves.shape[-1]))
        valid = torch.zeros((len(examples), kmax), dtype=torch.bool, device=device)
        offset = 0
        for i, ex in enumerate(examples):
            n = len(ex['leaf_tokens'])
            h[i, :n] = leaves[offset:offset+n]
            valid[i, :len(ex['candidate_ids'])] = True
            offset += n
        h = self.norm(h)
        z = self.scalar(h).squeeze(-1).float()
        choice = torch.tensor([i for i, ex in enumerate(examples) if ex['type'] == 'choice'], device=device)
        if self.set_head == 'attention' and len(choice):
            log_k = valid[choice].sum(-1).float().log()[:, None, None].expand(-1, kmax, 1)
            u = self.set_project(torch.cat([h[choice], log_k.to(h.dtype)], dim=-1))
            mixed, _ = self.set_attention(u, u, u, key_padding_mask=~valid[choice], need_weights=False)
            delta = self.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
            z = z.index_add(0, choice, delta)
        # Boolean has one semantic path and one scalar, representing logits [0,z].
        out = []
        for i, ex in enumerate(examples):
            if ex['type'] == 'boolean':
                out.append(F.pad(torch.stack([z[i, 0] * 0, z[i, 0]]), (0, kmax-2)))
            else:
                out.append(z[i])
        return torch.stack(out).masked_fill(~valid, -1e9), valid


def load_decision_model_class():
    # DecisionModel is inlined here to keep the vendored module self-contained.
    return DecisionModel


class DecisionPredictor:
    """Load local weights once and batch complete questions for prediction."""

    def __init__(self, checkpoint_dir, max_length=None, device_name="cuda:0",
                 disable_native_triton=False, precision="bf16"):
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        root, paths = local_checkpoint_files(checkpoint_dir)
        run_config = read_json(paths["run_config"])
        if not isinstance(run_config, dict) or run_config.get("set_head") not in {"none", "attention"}:
            raise ValueError("checkpoint config needs a valid set_head")
    
        # Load only the local checkpoint; do not fetch weights from the Hub.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    
        if disable_native_triton:
            from torch._native import triton_utils
            triton_utils.deregister_op_overrides()
        device = torch.device(device_name)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("This inference entry point requires a CUDA device")
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("The current CUDA device does not support BF16")
        torch.backends.cuda.matmul.allow_tf32 = False
    
        tokenizer = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), local_files_only=True,
                                                 trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        body_config = AutoConfig.from_pretrained(str(paths["body_config"]), local_files_only=True,
                                                trust_remote_code=False)
        body_config.use_cache = False
        limit = run_config.get("max_length", 512) if max_length is None else max_length
        if type(limit) is not int or limit <= 0:
            raise ValueError("max-length must be a positive integer")
        context_limit = getattr(body_config, "max_position_embeddings", None)
        if isinstance(context_limit, int) and limit > context_limit:
            raise ValueError("max-length exceeds the backbone context limit")
    
        # Build the architecture from config; load all weights from best.safetensors.
        body = AutoModel.from_config(body_config, attn_implementation="sdpa", trust_remote_code=False).float()
        DecisionModel = load_decision_model_class()
        model = DecisionModel(body, run_config["set_head"])
        weights = load_file(str(paths["weights"]), device="cpu")
        model.load_state_dict(weights, strict=True)
        del weights
        model.to(device=device, dtype=torch.float32)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.root = root
        self.run_config = run_config
        self.limit = limit
        self.device = device
        self.precision = precision
        self.disable_native_triton = disable_native_triton
        self.inference_calls = 0
        self._torch = torch

    def predict(self, payload, batch_questions=0, temperature=1.0):
        states = validate_request(payload)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        torch = self._torch
        model, tokenizer = self.model, self.tokenizer
        root, run_config, limit = self.root, self.run_config, self.limit
        device, precision = self.device, self.precision
        disable_native_triton = self.disable_native_triton
        examples = prepare_examples(payload, tokenizer, limit)
        batches = complete_question_batches(examples, batch_questions)
        self.inference_calls += 1
        model.eval()
        outputs = {state["id"]: {"id": state["id"], "answers": {}} for state in states}
        with torch.inference_mode():
            for batch in batches:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                    logits, _ = model(batch, tokenizer.pad_token_id)
                for example, values in zip(batch, logits):
                    k = len(example["candidate_ids"])
                    scores = values[:k].float()
                    if not torch.isfinite(scores).all():
                        raise ValueError("Model produced non-finite logits")
                    probabilities = (scores / temperature).softmax(-1).cpu().tolist()
                    outputs[example["state_id"]]["answers"][example["qid"]] = answer_from_probabilities(example, probabilities)
        return {
            "schema_version": "openjev-toy-inference-v1",
            "checkpoint": {"directory": str(root), "base_model": run_config.get("model"),
                           "base_revision": run_config.get("resolved_model_revision"), "set_head": run_config["set_head"]},
            "temperature": {"value": float(temperature), "fitted_by_this_command": False,
                            "note": "The specified scale is applied; 1.0 does not imply calibration."},
            "execution": {"device": str(device), "parameter_storage": "float32", "precision": precision,
                          "forward_autocast": "bfloat16" if precision == "bf16" else "disabled",
                          "states": len(states), "questions": len(examples),
                          "candidate_paths": sum(len(ex["leaf_tokens"]) for ex in examples),
                          "forward_passes": len(batches), "batch_questions_limit": batch_questions or "all",
                          "autoregressive_decode_steps": 0, "prefix_sharing": False,
                          "max_length": limit, "disable_native_triton": disable_native_triton,
                          "network_model_calls": 0, "persistent_model_load_count": 1,
                          "inference_call_index": self.inference_calls},
            "states": list(outputs.values()),
        }


def predict(payload, checkpoint_dir, temperature=1.0, batch_questions=0, max_length=None,
            device_name="cuda:0", disable_native_triton=False, precision="bf16"):
    """One-shot interface; reuse DecisionPredictor for repeated calls."""
    # Fail on malformed input before loading a checkpoint, as in the original entry point.
    validate_request(payload)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    engine = DecisionPredictor(checkpoint_dir, max_length=max_length, device_name=device_name,
                               disable_native_triton=disable_native_triton, precision=precision)
    return engine.predict(payload, batch_questions=batch_questions, temperature=temperature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True, help="JSON file containing a states array")
    parser.add_argument("--output", help="write output here, or to stdout when omitted")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-questions", type=int, default=0, help="0 batches all questions")
    parser.add_argument("--max-length", type=int, help="default: checkpoint config; oversized input raises an error")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16",
                        help="bf16 is the training default; fp32 disables autocast")
    parser.add_argument("--disable-native-triton", action="store_true", help="use the ATen fallback")
    args = parser.parse_args()
    try:
        result = predict(read_json(args.input), args.checkpoint_dir, args.temperature, args.batch_questions,
                         args.max_length, args.device, args.disable_native_triton, precision=args.precision)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            print(json.dumps({"output": str(destination), "execution": result["execution"]}, ensure_ascii=False))
        else:
            print(text, end="")
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
