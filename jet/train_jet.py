#!/usr/bin/env python3
"""Jet: fully-unified latent-space episodic memory for Jev-class decision models.

Jet is the final Long-Jev training method (development name C7): a single
persistent memory tier -- compressed latent slots only, no raw recent-step
retention -- written by a learned latent write head. The write head forwards
free continuous slot embeddings over the cache (in-distribution base K/V) and
adds a gated low-rank delta projected directly from the last hidden state,
bypassing the tokenizer; gates and slot scales start at zero, so training
begins exactly at C4-style behavior and learns to write pure latent content.
Perception stays token-based; memory is pure latent. Raw step K/V lives only
transiently inside the write window and is evicted at every boundary.

The ablation family lives in train_mem_generic.py:
  c3  linear transcript KV (windowed BPTT)
  c4  two-tier bounded memory, fixed-token notes
  c5  unified compression, fixed-token notes (structure-only unification)
  c6  two-tier, latent write head (mechanism-only unification)
  c7 == Jet (canonical entry point: this file)
"""
import sys

from jet.train_mem_generic import main

if __name__ == "__main__":
    if "--arm" in sys.argv:
        sys.exit("train_jet.py trains Jet (c7) only; use train_mem_generic.py "
                 "for the ablation arms c3-c6")
    sys.argv += ["--arm", "c7"]
    main()
