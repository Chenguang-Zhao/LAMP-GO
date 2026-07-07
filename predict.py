#!/usr/bin/env python3
"""Predict GO terms for a set of proteins with a trained LAMP-GO checkpoint.

    python predict.py --data-dir data --weights trained_lamp_go.pt --split test

Outputs (into --out-dir):
    <split>_scores.npy   float32 [n_proteins, 6416] predicted probabilities
    <split>.tsv          CAFA submission: EntryID <TAB> GO_term <TAB> score
"""
import argparse
import os

import numpy as np
import torch

from data import load_split, load_terms, make_batches, write_submission
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--weights", default="trained_lamp_go.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, tokenize = build_model(DEVICE)
    ckpt = torch.load(args.weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
    model.eval()
    print(f"loaded {args.weights}: {len(ckpt['state_dict'])} trained tensors", flush=True)

    ids, seqs, _ = load_split(args.data_dir, args.split)
    terms = load_terms(args.data_dir)
    batches = make_batches([len(s) for s in seqs])
    print(f"{args.split}: {len(ids)} proteins in {len(batches)} batches", flush=True)

    scores = np.empty((len(ids), len(terms)), dtype=np.float32)
    with torch.no_grad():
        for bi, idx in enumerate(batches):
            toks = tokenize([seqs[i] for i in idx]).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(toks)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for k, i in enumerate(idx):
                scores[i] = probs[k]
            if bi % 200 == 0:
                print(f"  batch {bi}/{len(batches)}", flush=True)

    npy = os.path.join(args.out_dir, f"{args.split}_scores.npy")
    tsv = os.path.join(args.out_dir, f"{args.split}.tsv")
    np.save(npy, scores)
    write_submission(ids, scores, tsv, terms)
    print(f"wrote {npy} and {tsv}", flush=True)


if __name__ == "__main__":
    main()
