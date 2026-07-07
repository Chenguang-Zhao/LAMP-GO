#!/usr/bin/env python3
"""Turn the raw CAFA-5 Kaggle download into the files LAMP-GO trains on.

Download the CAFA-5 data from Kaggle
(https://www.kaggle.com/competitions/cafa-5-protein-function-prediction) and
point --kaggle-dir at the unzipped folder. It must contain:

    Train/train_sequences.fasta            protein sequences
    Train/train_terms.tsv                  EntryID, term, aspect  (GO annotations)
    Train/train_taxonomy.tsv               EntryID, taxonomyID
    IA.txt                                 information accretion per GO term
    Test/testsuperset.fasta                test sequences (prediction target)

This writes into --out-dir (default ./data):

    candidate_terms.txt   one GO id per line, ordered by column index (6416 terms)
    train.csv             EntryID, sequence, terms   (space-joined GO ids)
    valid.csv             EntryID, sequence, terms
    test.csv              EntryID, sequence          (no labels -- prediction target)
    train_labels.npz      scipy CSR uint8  [n_train, 6416], row-aligned to train.csv
    valid_labels.npz      scipy CSR uint8  [n_valid, 6416], row-aligned to valid.csv

Definitions
    candidate vocabulary : GO terms annotated at least MIN_FREQ times in the train
        pool, minus the 3 GO-DAG roots. Annotations are already propagated to the
        root, so this set is ancestor-closed.
    train / valid split  : a per-protein 90/10 split with a fixed seed, so it is
        reproducible. CAFA-5 is temporal (test proteins are post-cutoff novelties),
        so no test proteins are removed from train.
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

GO_ROOTS = {"GO:0008150", "GO:0005575", "GO:0003674"}   # BPO / CCO / MFO roots
SEED = 42
VALID_FRAC = 0.10
MIN_FREQ = 50          # a GO term enters the vocabulary if annotated >= this often


def read_fasta(path):
    """Yield (id, sequence) for each record in a FASTA file."""
    rid, chunks = None, []
    for line in open(path):
        line = line.replace("\r", "").rstrip("\n")
        if line.startswith(">"):
            if rid is not None:
                yield rid, "".join(chunks)
            rid, chunks = line[1:].split()[0], []
        elif line:
            chunks.append(line)
    if rid is not None:
        yield rid, "".join(chunks)


def build_labels(df, term2idx):
    """CSR 0/1 matrix [n_rows, n_terms] from the space-joined `terms` column."""
    rows, cols = [], []
    for r, terms in enumerate(df.terms.fillna("")):
        for go in terms.split():
            j = term2idx.get(go)
            if j is not None:
                rows.append(r)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                         shape=(len(df), len(term2idx)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-dir", required=True, help="unzipped CAFA-5 Kaggle folder")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- candidate GO vocabulary ------------------------------------------
    terms = pd.read_csv(os.path.join(args.kaggle_dir, "Train/train_terms.tsv"), sep="\t")
    freq = terms.term.value_counts()
    aspect_of = dict(zip(terms.term, terms.aspect))
    keep = sorted(t for t, c in freq.items() if c >= MIN_FREQ and t not in GO_ROOTS)
    vocab = pd.DataFrame({"term": keep})
    vocab["aspect"] = vocab.term.map(aspect_of)
    vocab = vocab.sort_values(["aspect", "term"]).reset_index(drop=True)
    term2idx = {t: i for i, t in enumerate(vocab.term)}
    with open(os.path.join(args.out_dir, "candidate_terms.txt"), "w") as fh:
        fh.write("\n".join(vocab.term) + "\n")
    print(f"candidate vocabulary: {len(vocab)} GO terms")

    # ---- per-protein sequences, taxonomy, and vocab-filtered term lists ----
    seqs = dict(read_fasta(os.path.join(args.kaggle_dir, "Train/train_sequences.fasta")))
    d = terms[terms.term.isin(term2idx)].copy()
    d["idx"] = d.term.map(term2idx)
    d = d.sort_values(["EntryID", "idx"])
    protein_terms = d.groupby("EntryID").term.apply(" ".join).to_dict()

    # ---- reproducible per-protein train / valid split ---------------------
    ids = np.array(sorted(seqs))
    np.random.default_rng(SEED).shuffle(ids)
    n_valid = round(len(ids) * VALID_FRAC)
    valid_ids = set(ids[:n_valid])

    for name, id_list in [("train", [i for i in ids if i not in valid_ids]),
                          ("valid", sorted(valid_ids))]:
        rows = [(i, seqs[i], protein_terms.get(i, "")) for i in sorted(id_list)]
        df = pd.DataFrame(rows, columns=["EntryID", "sequence", "terms"])
        df.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        mat = build_labels(df, term2idx)
        sp.save_npz(os.path.join(args.out_dir, f"{name}_labels.npz"), mat)
        print(f"  {name}: {len(df)} proteins, {mat.nnz} positive labels")

    # ---- test: sequences only, no labels (the prediction target) ----------
    test_rows = sorted(read_fasta(os.path.join(args.kaggle_dir, "Test/testsuperset.fasta")))
    pd.DataFrame(test_rows, columns=["EntryID", "sequence"]) \
        .to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    print(f"  test: {len(test_rows)} proteins (no labels)")
    print("done")


if __name__ == "__main__":
    main()
