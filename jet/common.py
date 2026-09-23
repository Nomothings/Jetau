"""Text templates, tokenization, and model loading for Jeτ."""
from pathlib import Path

def leaf_text(key, description):
    return f"Candidate:\n{key}: {description}\nDecision:"


def tokenize_segments(tokenizer, segments):
    ids = []
    for text in segments:
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
    return ids


def load_decision_model(checkpoint_dir, device="cuda:0", mem_fraction=None):
    import torch
    if mem_fraction:
        torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)
    from jet.vendor.predict_toy_decisions import DecisionPredictor
    runtime = DecisionPredictor(checkpoint_dir, device_name=device)
    return runtime


def save_bundle(model, tokenizer, source_dir, out_dir):
    import shutil
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(source_dir) / "config.json", out_dir / "config.json")
    shutil.copytree(Path(source_dir) / "tokenizer", out_dir / "tokenizer", dirs_exist_ok=True)
    shutil.copytree(Path(source_dir) / "backbone_config", out_dir / "backbone_config", dirs_exist_ok=True)
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
              out_dir / "best.safetensors")
