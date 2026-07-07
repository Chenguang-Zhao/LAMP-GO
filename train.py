#!/usr/bin/env python3
"""Train LAMP-GO: LoRA fine-tune ESM-C 600M with a layer-mix + attention-pool head.

Only the LoRA adapters, layer-mix, attention pool, and 2-layer head are trained;
the ESM-C backbone stays frozen. Training uses bf16 autocast and gradient
checkpointing so the 600M backbone fits on a single 24 GB GPU. It early-stops on
validation loss and saves the best epoch's weights.

    python train.py --data-dir data --out lamp_go.pt

Outputs:
    lamp_go.pt          {"state_dict", "config"} -- the trained weights (~78 MB)
    lamp_go.history.json  per-epoch train/val loss and validation micro-Fmax
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from data import load_split, load_terms, make_batches
from model import (BACKBONE, D_MODEL, HIDDEN, ATTN_DIM, DROPOUT, N_TERMS,
                   LORA_R, LORA_ALPHA, LORA_TARGETS, build_model, trainable_state)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR_LORA = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 30         # a high cap; early stopping on validation loss ends training
PATIENCE = 3
SEED = 42


def enable_grad_checkpointing(esmc):
    """Recompute each transformer block in the backward pass to save activation memory."""
    for blk in esmc.transformer.blocks:
        orig = blk.forward
        blk.forward = (lambda f: lambda *a, **k: cp.checkpoint(f, *a, use_reentrant=False, **k))(orig)


@torch.no_grad()
def micro_fmax(probs, Y):
    """Micro-averaged Fmax over a threshold sweep -- the CAFA-5 primary metric."""
    best = 0.0
    for th in np.linspace(0.05, 0.95, 19):
        pred = probs >= th
        tp = np.logical_and(pred, Y > 0.5).sum()
        fp = np.logical_and(pred, Y < 0.5).sum()
        fn = np.logical_and(~pred, Y > 0.5).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        best = max(best, 2 * p * r / (p + r + 1e-9))
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="lamp_go.pt")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device={DEVICE}", flush=True)

    model, tokenize = build_model(DEVICE)
    enable_grad_checkpointing(model.esmc.base_model.model)
    model.esmc.print_trainable_parameters()

    tr_ids, tr_seqs, Ytr = load_split(args.data_dir, "train")
    va_ids, va_seqs, Yva = load_split(args.data_dir, "valid")
    tr_batches = make_batches([len(s) for s in tr_seqs])
    va_batches = make_batches([len(s) for s in va_seqs])
    print(f"train {len(tr_seqs)} ({len(tr_batches)} batches)  "
          f"valid {len(va_seqs)} ({len(va_batches)} batches)  -> {N_TERMS} GO terms", flush=True)

    opt = torch.optim.AdamW([
        {"params": [p for p in model.esmc.parameters() if p.requires_grad],
         "lr": LR_LORA, "weight_decay": WEIGHT_DECAY},
        {"params": model.head.parameters(), "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
        {"params": model.pool.parameters(), "lr": LR_HEAD, "weight_decay": 0.0},
        {"params": model.mix.parameters(), "lr": LR_HEAD, "weight_decay": 0.0},
    ])
    bce = nn.BCEWithLogitsLoss()

    def run_epoch(batches, seqs, Y, train):
        model.train(train)
        order = np.random.permutation(len(batches)) if train else range(len(batches))
        total, seen = 0.0, 0
        logits_out = None if train else np.empty((len(seqs), N_TERMS), dtype=np.float32)
        for bi in order:
            idx = batches[bi]
            toks = tokenize([seqs[i] for i in idx]).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(toks)
                loss = bce(logits, torch.from_numpy(Y[idx]).to(DEVICE))
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(idx)
            seen += len(idx)
            if logits_out is not None:
                logits_out[idx] = logits.float().detach().cpu().numpy()
        return total / seen, logits_out

    best_val, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _ = run_epoch(tr_batches, tr_seqs, Ytr, train=True)
        with torch.no_grad():
            val_loss, vlogits = run_epoch(va_batches, va_seqs, Yva, train=False)
        vf = micro_fmax(1 / (1 + np.exp(-vlogits)), Yva)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_microF": vf})
        print(f"epoch {epoch:2d}  train {train_loss:.5f}  val {val_loss:.5f}  "
              f"val_microF {vf:.4f}", flush=True)

        if val_loss < best_val - 1e-5:
            best_val, bad, best_state = val_loss, 0, trainable_state(model)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    config = {"model": "LAMP-GO", "backbone": BACKBONE, "n_terms": N_TERMS,
              "d_model": D_MODEL, "hidden": HIDDEN, "attn_dim": ATTN_DIM,
              "dropout": DROPOUT, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
              "lora_targets": LORA_TARGETS, "best_val_loss": best_val,
              "epochs_run": len(history), "seed": SEED}
    torch.save({"state_dict": best_state, "config": config}, args.out)
    with open(args.out.replace(".pt", "") + ".history.json", "w") as fh:
        json.dump({"config": config, "history": history}, fh, indent=2)
    print(f"saved {args.out} (best val loss {best_val:.5f})", flush=True)


if __name__ == "__main__":
    main()
